#!/usr/bin/env python3
"""CSV Motion / Flow Replay Viewer
=================================
Plays back a recorded video with motion or flow data from a companion CSV
overlaid on top.

Usage:
    python3 csv_replay.py [video_or_csv_path] [--csv CSV]

The first positional argument may be either a video file or a CSV file:
  - Video: companion CSV is auto-detected (_deepflow, _dense_flow, _motion, or _offline_motion suffixes)
  - CSV:   companion video is auto-detected by stripping the CSV suffix

If no path is given a file dialog opens.
Use --csv to supply an explicit CSV regardless of the first argument.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Some flow CSVs have very long lines — raise the field-size limit to max
csv.field_size_limit(sys.maxsize)

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSortFilterProxyModel, QTimer
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ─── File-dialog proxy: hide videos that have no companion CSV ──────────────

class _CsvPairedFilter(QSortFilterProxyModel):
    """Only show directories and video files that have a companion CSV."""
    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        if model.isDir(index):
            return True
        p = Path(model.filePath(index))
        return (p.with_name(p.stem + "_dense_flow.csv").is_file()
                or p.with_name(p.stem + "_motion.csv").is_file())


# ─── CSV loading ────────────────────────────────────────────────────────────

def _detect_csv_format(csv_path: str) -> str:
    """Peek at header to decide format: 'dense_flow' or 'motion'."""
    with open(csv_path, newline="") as f:
        header = csv.DictReader(f).fieldnames or []
    if "gx" in header and "gy" in header:
        return "dense_flow"
    return "motion"


def _load_dense_flow_csv(csv_path: str) -> dict[int, list[dict]]:
    """Load a _dense_flow.csv → {frame: [{type, gx, gy, dx, dy, mag}, ...]}."""
    data: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            data[frame].append({
                "type": "flow",
                "gx": int(row["gx"]),
                "gy": int(row["gy"]),
                "dx": float(row["dx"]),
                "dy": float(row["dy"]),
                "magnitude": float(row["mag"]),
            })
    return dict(data)


def _load_motion_csv(csv_path: str) -> dict[int, list[dict]]:
    """Load a _motion.csv → {frame: [row_dict, ...]}."""
    data: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            entry: dict = {
                "type": row["type"],
                "track_id": int(row["track_id"]) if row.get("track_id", "") not in ("", "-1") else -1,
                "bbox": (
                    int(row["bbox_x"]),
                    int(row["bbox_y"]),
                    int(row["bbox_w"]),
                    int(row["bbox_h"]),
                ),
            }
            if row["type"] == "vector_blob":
                cog_x = row.get("cog_x", "")
                cog_y = row.get("cog_y", "")
                entry["cog"] = (
                    float(cog_x) if cog_x else None,
                    float(cog_y) if cog_y else None,
                )
                mag = row.get("magnitude", "")
                entry["magnitude"] = float(mag) if mag else 0.0
            data[frame].append(entry)
    return dict(data)


def load_csv(csv_path: str) -> tuple[dict[int, list[dict]], str]:
    """Load CSV, auto-detecting format.  Returns (data, format_name)."""
    fmt = _detect_csv_format(csv_path)
    if fmt == "dense_flow":
        return _load_dense_flow_csv(csv_path), fmt
    return _load_motion_csv(csv_path), fmt


# ─── Drawing ────────────────────────────────────────────────────────────────

DET_COLOR = (0, 255, 0)       # green  (BGR)
VEC_COLOR = (255, 255, 0)     # cyan   (BGR)
COG_COLOR = (0, 0, 255)       # red    (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 2


FLOW_ARROW_SCALE = 3.0


def draw_overlays(frame: np.ndarray, rows: list[dict],
                  csv_format: str = "motion") -> np.ndarray:
    """Draw CSV entries onto a BGR frame (mutates in-place)."""
    if csv_format == "dense_flow":
        return _draw_flow_overlays(frame, rows)
    return _draw_motion_overlays(frame, rows)


def _draw_flow_overlays(frame: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Draw dense_flow vectors as direction-coloured arrows."""
    for r in rows:
        gx, gy = r["gx"], r["gy"]
        dx, dy = r["dx"], r["dy"]
        mag = r["magnitude"]
        # Direction → hue
        angle = np.arctan2(dy, dx)
        hue = int(((angle + np.pi) / (2 * np.pi)) * 179)
        val = min(255, int(mag * 20 + 80))
        hsv_pixel = np.array([[[hue, 255, val]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        color = tuple(int(c) for c in bgr_pixel[0, 0])
        tip_x = int(gx + dx * FLOW_ARROW_SCALE)
        tip_y = int(gy + dy * FLOW_ARROW_SCALE)
        cv2.arrowedLine(frame, (gx, gy), (tip_x, tip_y), color, 1,
                        tipLength=0.3)
    return frame


def _draw_motion_overlays(frame: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Draw original motion CSV entries (bboxes, vector blobs)."""
    for r in rows:
        x, y, w, h = r["bbox"]
        is_vec = r["type"] == "vector_blob"
        color = VEC_COLOR if is_vec else DET_COLOR

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, THICKNESS)

        # Label: track ID
        tid = r["track_id"]
        label_parts: list[str] = []
        if tid >= 0:
            label_parts.append(f"ID:{tid}")
        if is_vec and r.get("magnitude"):
            label_parts.append(f"mag:{r['magnitude']:.1f}")
        if label_parts:
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            lx, ly = x, y - 6
            if ly - th < 0:
                ly = y + h + th + 4
            cv2.rectangle(frame, (lx - 1, ly - th - 2), (lx + tw + 2, ly + 3), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (lx, ly), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)

        # CoG marker for vector blobs
        if is_vec and r.get("cog"):
            cx, cy = r["cog"]
            if cx is not None and cy is not None:
                cv2.drawMarker(frame, (int(cx), int(cy)), COG_COLOR, cv2.MARKER_CROSS, 14, 2)

    return frame


# ─── Qt viewer ──────────────────────────────────────────────────────────────

class ReplayWindow(QMainWindow):
    def __init__(self, video_path: str, csv_data: dict[int, list[dict]],
                 csv_format: str = "motion"):
        super().__init__()
        self.setWindowTitle(f"CSV Replay — {Path(video_path).name}")
        self._csv_format = csv_format

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._csv = csv_data
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._current_frame = 0
        self._playing = False

        self._init_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        # Show first frame
        self._seek(0)

    # ── UI ───────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display
        self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(640, 360)
        self._label.setStyleSheet("background: black;")
        layout.addWidget(self._label, stretch=1)

        # Transport bar
        transport = QHBoxLayout()
        layout.addLayout(transport)

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(60)
        self._play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, self._total_frames - 1))
        self._slider.sliderPressed.connect(self._slider_pressed)
        self._slider.sliderReleased.connect(self._slider_released)
        self._slider.valueChanged.connect(self._slider_changed)
        transport.addWidget(self._slider, stretch=1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(120)
        transport.addWidget(self._frame_label)

        # Speed control
        speed_lbl = QLabel("Speed:")
        transport.addWidget(speed_lbl)
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 40)  # 0.1x to 4.0x
        self._speed_slider.setValue(10)      # 1.0x
        self._speed_slider.setFixedWidth(100)
        self._speed_slider.valueChanged.connect(self._speed_changed)
        transport.addWidget(self._speed_slider)
        self._speed_label = QLabel("1.0x")
        self._speed_label.setFixedWidth(40)
        transport.addWidget(self._speed_label)

        # Keyboard shortcuts
        play_action = QAction("Play/Pause", self)
        play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        play_action.triggered.connect(self._toggle_play)
        self.addAction(play_action)

        left_action = QAction("Back", self)
        left_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        left_action.triggered.connect(lambda: self._step(-1))
        self.addAction(left_action)

        right_action = QAction("Forward", self)
        right_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        right_action.triggered.connect(lambda: self._step(1))
        self.addAction(right_action)

        self.resize(1024, 640)

    # ── Playback ─────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self._playing = True
        self._play_btn.setText("Pause")
        speed = self._speed_slider.value() / 10.0
        self._timer.start(int(1000.0 / (self._fps * speed)))

    def _pause(self):
        self._playing = False
        self._play_btn.setText("Play")
        self._timer.stop()

    def _speed_changed(self, val):
        speed = val / 10.0
        self._speed_label.setText(f"{speed:.1f}x")
        if self._playing:
            self._timer.setInterval(int(1000.0 / (self._fps * speed)))

    def _step(self, delta: int):
        self._pause()
        target = max(0, min(self._current_frame + delta, self._total_frames - 1))
        self._seek(target)

    def _next_frame(self):
        if self._current_frame >= self._total_frames - 1:
            self._pause()
            return
        self._show_frame(self._current_frame + 1)

    def _seek(self, frame_no: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self._show_frame(frame_no)

    def _show_frame(self, frame_no: int):
        # If cap position doesn't match (e.g. after seek) make sure we read
        # from the right place.
        if int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_no:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, bgr = self._cap.read()
        if not ret:
            self._pause()
            return

        self._current_frame = frame_no

        # Draw CSV overlays
        if frame_no in self._csv:
            draw_overlays(bgr, self._csv[frame_no], self._csv_format)

        # Draw frame counter
        text = f"Frame {frame_no}"
        cv2.putText(bgr, text, (10, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (10, 30), FONT, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        # Convert BGR → RGB → QImage → QPixmap
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label while keeping aspect ratio
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

        # Update slider and label (block signals to avoid seek loop)
        self._slider.blockSignals(True)
        self._slider.setValue(frame_no)
        self._slider.blockSignals(False)
        self._frame_label.setText(f"{frame_no} / {self._total_frames - 1}")

    # ── Slider interaction ───────────────────────────────────────────
    def _slider_pressed(self):
        self._was_playing = self._playing
        if self._playing:
            self._pause()

    def _slider_released(self):
        self._seek(self._slider.value())
        if getattr(self, "_was_playing", False):
            self._play()

    def _slider_changed(self, val):
        if self._slider.isSliderDown():
            self._seek(val)

    # ── Cleanup ──────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._timer.stop()
        self._cap.release()
        super().closeEvent(event)


# ─── Main ───────────────────────────────────────────────────────────────────

def _find_csv(video: Path, explicit: str | None = None) -> tuple[Path | None, str]:
    """Locate companion CSV.  Returns (path, format) or (None, '')."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p, _detect_csv_format(str(p))
        return None, ""
    # Prefer deepflow, dense_flow, then motion, then _recording variants
    for suffix, fmt in [
        ("_deepflow.csv", "dense_flow"),
        ("_dense_flow.csv", "dense_flow"),
        ("_motion.csv", "motion"),
        ("_offline_motion.csv", "motion"),
    ]:
        candidate = video.with_name(video.stem + suffix)
        if candidate.is_file():
            return candidate, fmt
    # Try stripping _recording suffix
    alt_stem = video.stem.replace("_recording", "")
    for suffix, fmt in [
        ("_dense_flow.csv", "dense_flow"),
        ("_motion.csv", "motion"),
    ]:
        candidate = video.with_name(alt_stem + suffix)
        if candidate.is_file():
            return candidate, fmt
    return None, ""


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MKV", ".AVI", ".MOV"}
_CSV_SUFFIXES = ["_deepflow", "_dense_flow", "_motion", "_offline_motion", "_recording_deepflow",
                 "_recording_dense_flow", "_recording_motion"]


def _find_video(csv_path: Path) -> Path | None:
    """Given a CSV path, locate the companion video by stripping known suffixes."""
    stem = csv_path.stem  # e.g. "VID_020_deepflow"
    # Strip known CSV suffixes to recover the video stem
    base_stem = stem
    for sfx in _CSV_SUFFIXES:
        if stem.endswith(sfx):
            base_stem = stem[: -len(sfx)]
            break
    directory = csv_path.parent
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MKV", ".AVI", ".MOV"]:
        candidate = directory / (base_stem + ext)
        if candidate.is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="CSV Motion/Flow Replay Viewer")
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to video file OR CSV file")
    parser.add_argument("--csv", metavar="CSV",
                        help="Explicit CSV file (auto-detected if omitted)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    explicit_csv: str | None = args.csv
    video_path: str | None = None

    if args.path:
        p = Path(args.path)
        if not p.is_file():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() == ".csv":
            # CSV supplied as first arg — locate companion video
            if explicit_csv is None:
                explicit_csv = str(p)
            found_video = _find_video(p)
            if found_video:
                video_path = str(found_video)
                print(f"Auto-detected video: {found_video}")
            else:
                print(f"Warning: no companion video found for {p.name}; opening file dialog")
        else:
            video_path = str(p)

    if video_path is None:
        dialog = QFileDialog(
            None,
            "Select Recorded Video — only videos with a matching CSV are shown",
            str(Path(__file__).parent / "rec"),
        )
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setNameFilter("Video Files (*.mp4 *.mkv *.avi)")
        dialog.setProxyModel(_CsvPairedFilter())
        if not dialog.exec() or not dialog.selectedFiles():
            sys.exit(0)
        video_path = dialog.selectedFiles()[0]

    video = Path(video_path)
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    # Find companion CSV
    csv_path, csv_fmt = _find_csv(video, explicit_csv)

    if csv_path is not None:
        print(f"Loading CSV: {csv_path}  (format: {csv_fmt})")
        csv_data, csv_fmt = load_csv(str(csv_path))
        frames_with_data = len(csv_data)
        total_rows = sum(len(v) for v in csv_data.values())
        print(f"  {total_rows} entries across {frames_with_data} frames")
    else:
        print(f"Warning: no CSV found for {video.name}, playing video only")
        csv_data = {}
        csv_fmt = "motion"

    window = ReplayWindow(str(video), csv_data, csv_fmt)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
