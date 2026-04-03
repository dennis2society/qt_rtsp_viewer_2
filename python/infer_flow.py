#!/usr/bin/env python3
"""
infer_motion.py — Fast motion-field inference & overlay
=======================================================
Runs a trained FlowNet checkpoint on a video, drawing the predicted motion
field as a colour overlay on top of the original frames.  Optionally writes
an output video file.

Speed improvements over the old train_motion.py --infer path:
  · Batch processing: reads ahead N frames and runs them through the network
    in a single batched forward pass.
  · Half-precision (fp16) inference when a CUDA GPU is available.
  · Flow visualisation computed at network resolution then upscaled (instead
    of upscaling the raw flow field first).
  · Display and file-writing happen asynchronously in a background thread.

Usage
-----
  # Live preview (press Q / Esc to quit):
  python3 infer_motion.py video.mp4 checkpoints/best.pt

  # Write output video (no preview window):
  python3 infer_motion.py video.mp4 checkpoints/best.pt -o output.mp4

  # Both preview and output:
  python3 infer_motion.py video.mp4 checkpoints/best.pt -o output.mp4 --show
"""

import argparse
import math
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


# ─── Network (must match train_motion.py) ────────────────────────────────────

class _CBR(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )


class FlowNet(nn.Module):
    """6-ch input (prev+curr RGB) -> 2-ch flow (dx, dy), bounded +/-max_flow."""
    _CH = (32, 64, 128, 256)

    def __init__(self, max_flow: float = 30.0):
        super().__init__()
        self.max_flow = max_flow
        c = self._CH
        self.e0 = _CBR(6, c[0])
        self.e1 = _CBR(c[0], c[1], stride=2)
        self.e2 = _CBR(c[1], c[2], stride=2)
        self.e3 = _CBR(c[2], c[3], stride=2)
        self.bot = nn.Sequential(_CBR(c[3], c[3]), _CBR(c[3], c[3]))
        self.d2 = nn.Sequential(_CBR(c[3] + c[2], c[2]), _CBR(c[2], c[2]))
        self.d1 = nn.Sequential(_CBR(c[2] + c[1], c[1]), _CBR(c[1], c[1]))
        self.d0 = nn.Sequential(_CBR(c[1] + c[0], c[0]), _CBR(c[0], c[0]))
        self.head  = nn.Conv2d(c[0], 2, 3, padding=1)
        # Multi-scale flow heads for deep supervision (1/2 and 1/4 res)
        self.head1 = nn.Conv2d(c[1], 2, 3, padding=1)
        self.head2 = nn.Conv2d(c[2], 2, 3, padding=1)

    def forward(self, prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prev, curr], dim=1)
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.bot(e3)

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:],
                                 mode="bilinear", align_corners=False)

        d2 = self.d2(torch.cat([up(b, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        d0 = self.d0(torch.cat([up(d1, e0), e0], 1))

        flow0 = torch.tanh(self.head(d0)) * self.max_flow
        if self.training:
            # Coarse flow predictions at lower resolutions.
            # Scale max_flow proportionally: pixel displacements are smaller
            # at lower spatial resolutions.
            flow1 = torch.tanh(self.head1(d1)) * (self.max_flow * 0.5)
            flow2 = torch.tanh(self.head2(d2)) * (self.max_flow * 0.25)
            return flow0, flow1, flow2
        return flow0


# ─── Flow → colour overlay ──────────────────────────────────────────────────

FLOW_ARROW_SCALE = 3.0  # Scale factor for flow vector arrows

def draw_flow_arrows(frame: np.ndarray, flow_np: np.ndarray, 
                     max_mag: float = 30.0) -> np.ndarray:
    """Draw dense flow vectors as direction-coloured arrows on frame.
    
    Args:
        frame: BGR image to draw on (H, W, 3)
        flow_np: flow vectors (2, H, W) with [dx, dy]
        max_mag: maximum magnitude for scaling
    
    Returns:
        Modified frame with arrows drawn
    """
    dx = flow_np[0]
    dy = flow_np[1]
    mag = np.hypot(dx, dy)
    
    # Sample points on a grid to avoid overcrowding
    h, w = dx.shape
    step = max(1, min(h, w) // 20)  # ~20x20 grid of arrows
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mag[y, x] < 0.5:  # Skip near-zero motion
                continue
            
            # Calculate arrow direction and color
            angle = np.arctan2(dy[y, x], dx[y, x])
            hue = int(((angle + math.pi) / (2 * math.pi)) * 179)
            
            # Brightness based on magnitude
            val = min(255, int(mag[y, x] * 20 + 80))
            
            # Convert HSV to BGR
            hsv_pixel = np.array([[[hue, 255, val]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            color = tuple(int(c) for c in bgr_pixel[0, 0])
            
            # Draw arrow from (x, y) to (x + dx*scale, y + dy*scale)
            tip_x = int(x + dx[y, x] * FLOW_ARROW_SCALE)
            tip_y = int(y + dy[y, x] * FLOW_ARROW_SCALE)
            cv2.arrowedLine(frame, (x, y), (tip_x, tip_y), color, 1,
                           tipLength=0.3)
    
    return frame


# ─── Video reader (reads ahead into a queue) ────────────────────────────────

class VideoReader:
    """Read frames from a video file in a background thread."""

    def __init__(self, path: str, queue_size: int = 64):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._q: Queue = Queue(maxsize=queue_size)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            ret, bgr = self._cap.read()
            if not ret:
                self._q.put(None)
                break
            self._q.put(bgr)
        self._cap.release()

    def read(self) -> np.ndarray | None:
        return self._q.get()


# ─── PyQt6 Display Window ───────────────────────────────────────────────────

class FrameSignaler(QObject):
    """Thread-safe frame signaler for PyQt6."""
    frame_ready = pyqtSignal(QImage)
    
    def emit_frame(self, bgr: np.ndarray):
        h, w, ch = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.frame_ready.emit(qt_image)


class DisplayWindow(QMainWindow):
    """PyQt6 window for displaying inference results."""
    
    def __init__(self, title: str = "Flow Inference"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1920, 1080)
        
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: black;")
        self.setCentralWidget(self._label)
        
        self._signaler = FrameSignaler()
        self._signaler.frame_ready.connect(self._on_frame_ready)
        
        self._quit_requested = False
    
    def _on_frame_ready(self, q_image: QImage):
        """Display frame from signal."""
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaledToHeight(self.height() - 50, Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)
    
    def display_frame(self, bgr: np.ndarray):
        """Display a BGR frame (thread-safe)."""
        self._signaler.emit_frame(bgr)
        QApplication.processEvents()
    
    def closeEvent(self, event):
        """Handle window close."""
        self._quit_requested = True
        event.accept()
    
    def keyPressEvent(self, event):
        """Handle keyboard input."""
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self._quit_requested = True
            self.close()


# ─── Main inference loop ────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and args.fp16

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    net_wh = tuple(ckpt["net_wh"])
    max_flow = ckpt.get("max_flow", 30.0)

    net = FlowNet(max_flow=max_flow).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    if use_fp16:
        net.half()
    print(f"Model   : epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}  "
          f"({'fp16' if use_fp16 else 'fp32'} on {device})")

    net_w, net_h = net_wh

    # Open video
    reader = VideoReader(str(args.video))
    orig_w, orig_h = reader.width, reader.height
    fps = reader.fps
    total = reader.total
    print(f"Video   : {orig_w}×{orig_h}, {total} frames @ {fps:.1f} fps")
    print(f"Network : {net_w}×{net_h}")

    # Output writer
    vw = None
    out_path = None
    if args.output:
        out_path = Path(args.output)
        vw = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (orig_w, orig_h),
        )
        print(f"Output  : {out_path}")

    # Display window
    show = args.show or (vw is None)
    display_window = None
    
    if show and PYQT_AVAILABLE:
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            display_window = DisplayWindow(f"Flow Inference — {Path(args.video).name}")
            # Don't call show() here - let it show on first frame
        except Exception as e:
            print(f"Warning: Could not initialize display ({e}), continuing without preview")
            display_window = None
    elif not PYQT_AVAILABLE and show:
        print("Warning: PyQt6 not available, skipping preview")

    # Scale factors for mag display
    max_mag = max_flow * 0.5
    batch_size = args.batch

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, unit="frame", desc="Inference",
                    dynamic_ncols=True)

    prev_t: torch.Tensor | None = None
    pending_bgrs: list[np.ndarray] = []
    frame_idx = 0
    done = False
    quit_early = False

    def to_tensor(bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
        if use_fp16:
            t = t.half()
        return t

    def process_batch(prev_tensors: list[torch.Tensor],
                      curr_tensors: list[torch.Tensor],
                      bgrs: list[np.ndarray],
                      start_idx: int) -> bool:
        """Run network on a batch, draw overlays, display/write. Returns False to quit."""
        prevs = torch.stack(prev_tensors).to(device)
        currs = torch.stack(curr_tensors).to(device)
        with torch.no_grad():
            flows = net(prevs, currs)  # (B, 2, net_h, net_w)
        flows_np = flows.float().cpu().numpy()

        for i in range(len(bgrs)):
            flow_np = flows_np[i]  # (2, net_h, net_w)

            # Upscale flow to original resolution
            flow_scaled = np.zeros((2, orig_h, orig_w), dtype=np.float32)
            for ch in range(2):
                flow_scaled[ch] = cv2.resize(flow_np[ch], (orig_w, orig_h),
                                             interpolation=cv2.INTER_LINEAR)
                # Scale the flow values by the resize factor
                if ch == 0:
                    flow_scaled[ch] *= orig_w / net_w
                else:
                    flow_scaled[ch] *= orig_h / net_h

            out_bgr = bgrs[i].copy()
            draw_flow_arrows(out_bgr, flow_scaled, max_mag)

            # Frame counter
            fidx = start_idx + i
            label = f"frame {fidx}"
            cv2.putText(out_bgr, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out_bgr, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 1, cv2.LINE_AA)

            if vw is not None:
                vw.write(out_bgr)

            if display_window is not None:
                if not display_window.isVisible():
                    display_window.show()
                display_window.display_frame(out_bgr)
                if display_window._quit_requested:
                    return False
        return True

    # Main read loop — accumulate frames into batches
    prev_tensors: list[torch.Tensor] = []
    curr_tensors: list[torch.Tensor] = []
    batch_bgrs: list[np.ndarray] = []
    batch_start_idx = 0

    t0 = time.monotonic()

    while not done and not quit_early:
        bgr = reader.read()
        if bgr is None:
            done = True
            # flush remaining batch
            if curr_tensors:
                quit_early = not process_batch(
                    prev_tensors, curr_tensors, batch_bgrs, batch_start_idx)
            break

        curr_t = to_tensor(bgr)

        if prev_t is not None:
            prev_tensors.append(prev_t)
            curr_tensors.append(curr_t)
            batch_bgrs.append(bgr)

            if len(curr_tensors) >= batch_size:
                quit_early = not process_batch(
                    prev_tensors, curr_tensors, batch_bgrs, batch_start_idx)
                if pbar:
                    pbar.update(len(curr_tensors))
                batch_start_idx = frame_idx + 1
                prev_tensors.clear()
                curr_tensors.clear()
                batch_bgrs.clear()

        prev_t = curr_t
        frame_idx += 1

    if pbar:
        pbar.update(pbar.total - pbar.n)
        pbar.close()

    elapsed = time.monotonic() - t0
    effective_fps = frame_idx / max(elapsed, 0.01)
    print(f"Done    : {frame_idx} frames in {elapsed:.1f}s "
          f"({effective_fps:.1f} fps)")

    if vw is not None:
        vw.release()
        print(f"Saved   : {out_path}")

    if display_window is not None:
        display_window.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=Path,
                   help="Input video file (.mp4/.mkv/.avi)")
    p.add_argument("checkpoint", type=Path,
                   help="FlowNet checkpoint (.pt)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output video path. If omitted, only live preview "
                        "is shown.")
    p.add_argument("--show", action="store_true",
                   help="Show live preview window even when writing output")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size for network inference (default 8)")
    p.add_argument("--blend", type=float, default=0.55,
                   help="Overlay blend strength 0-1 (default 0.55)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   default=True,
                   help="Disable half-precision inference on GPU")
    args = p.parse_args()

    if not args.video.is_file():
        sys.exit(f"Video not found: {args.video}")
    if not args.checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    run(args)


if __name__ == "__main__":
    main()
