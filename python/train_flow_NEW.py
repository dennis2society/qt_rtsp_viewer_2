#!/usr/bin/env python3
"""
train_flow_NEW.py -- Dense optical-flow trainer (fast JPEG-in-RAM cache)
=======================================================================
Trains a lightweight U-Net to predict dense optical flow from consecutive
video frames.  Same model architecture and losses as train_flow.py but with
a much faster data pipeline.

**Loading strategy -- JPEG-in-RAM frame cache**:
All video frames are decoded *once* sequentially (fast, no random seeks),
resized to the network input size, JPEG-compressed, and stored in a Python
list.  At training time, random-access decompression takes ~0.2 ms per frame
(vs ~5-20 ms per random VideoCapture seek).  JPEG compression at quality 95
reduces memory ~8-10x compared to raw uint8 arrays, so 100 k frames at
256x256 occupy ~2-3 GB of RAM instead of ~20 GB.

Usage:
    python train_flow_NEW.py /path/to/recordings/
    python train_flow_NEW.py /path/to/recordings/ --csv camera_list.csv
    python train_flow_NEW.py /path/to/recordings/ --export

Options:
    --csv FILE      Limit videos to those listed in CSV file (columns: video, csv)
    --val-split F   Validation split fraction (0.0-1.0, default: 0.1)
    --export        Export best.pt to ONNX format
"""

# ============================================================
#                        CONFIGURATION
# ============================================================

NET_SHORT_SIDE   = 256       # shorter side of net input; rounded to x8
STRIDE           = 1         # use every Nth video frame (1 = all)
BATCH_SIZE       = 16        # larger batch for stable gradients
EPOCHS           = 30
LR               = 3e-4
MAX_FLOW         = 30.0      # tanh output bound (pixels)
JPEG_QUALITY     = 95        # in-RAM JPEG compression quality (90-100)
NUM_WORKERS      = 0         # 0 = run in main process; avoids fork-copy of the
                             #  JPEG cache (each worker would get its own full copy
                             #  via copy-on-write, blowing RAM by 4x)
WARMUP_EPOCHS    = 3         # linear LR warm-up before cosine decay
VAL_SPLIT        = 0.1       # fraction of frames for validation (default)
SAVE_DIR         = "checkpoints_flow"
RESUME_PATH      = ""        # set to checkpoint path to resume training

# Loss weights
W_PHOTO          = 0.5       # photometric consistency (reduced: noisy with camera motion)
W_FLOW           = 8.0       # direct flow-vector supervision (primary signal)
W_BG             = 0.5       # background suppression
W_TV             = 0.05      # total-variation smoothness

# ============================================================

import gc
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}


# --- Helpers ----------------------------------------------------------------

def _net_size(orig_w: int, orig_h: int) -> tuple[int, int]:
    """Network input size: scale so shorter side approx NET_SHORT_SIDE, x8."""
    s  = NET_SHORT_SIDE / min(orig_w, orig_h)
    nw = round(orig_w * s / 8) * 8
    nh = round(orig_h * s / 8) * 8
    return nw, nh


def _actual_frame_count(cap: cv2.VideoCapture, reported: int) -> int:
    """Scan backward to find the last truly readable frame."""
    if reported <= 0:
        return 0
    for delta in range(min(64, reported)):
        idx = reported - 1 - delta
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, _ = cap.read()
        if ret:
            return idx + 1
    return reported


# --- CSV loader ---------------------------------------------------------------

def load_csv(csv_path: str) -> dict[int, np.ndarray]:
    """Return {frame_index: ndarray(N, 5)} columns [gx, gy, dx, dy, mag]."""
    _COLS = ["gx", "gy", "dx", "dy", "mag"]
    _DTYPES = {"frame": np.int32, "gx": np.int32, "gy": np.int32,
               "dx": np.float32, "dy": np.float32, "mag": np.float32}
    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    for chunk in pd.read_csv(csv_path, usecols=["frame"] + _COLS,
                             dtype=_DTYPES, chunksize=500_000):
        for fi, grp in chunk.groupby("frame", sort=False):
            accum[int(fi)].append(grp[_COLS].to_numpy(dtype=np.float32))
    return {k: (np.concatenate(v) if len(v) > 1 else v[0])
            for k, v in accum.items()}


def _find_csv(video: Path) -> Path | None:
    dense = video.with_name(video.stem + "_dense_flow.csv")
    return dense if dense.is_file() else None


def load_video_list(csv_path: str) -> set[str]:
    """Load video names from CSV file (expects 'video' column)."""
    try:
        df = pd.read_csv(csv_path, usecols=["video"])
        video_names = set(str(v).strip() for v in df["video"].values)
        print(f"Loaded {len(video_names)} videos from {csv_path}")
        return video_names
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return set()


def _collect_pairs(root: Path, video_filter: set[str] | None = None) -> list[tuple[Path, dict[int, np.ndarray]]]:
    """Discover video+CSV pairs and load CSV annotations.
    
    If video_filter is provided, only include videos whose basename is in the set.
    """
    videos = sorted(p for p in root.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"No video files in: {root}")

    if video_filter is not None:
        videos = [v for v in videos if v.name in video_filter]
        if not videos:
            sys.exit(f"No videos in the filter list found in: {root}")

    pairs = []
    for v in videos:
        csv_p = _find_csv(v)
        if csv_p is None:
            print(f"  skip : {v.name}  (no _dense_flow.csv)")
            continue
        print(f"  +    : {v.name}  ← {csv_p.name}")
        pairs.append((v, load_csv(str(csv_p))))

    if not pairs:
        sys.exit("No videos with a companion _dense_flow.csv found.")
    return pairs


# --- JPEG-in-RAM frame cache -------------------------------------------------

class FrameCache:
    """
    Stores all video frames as JPEG-compressed bytes in RAM.

    Build phase (sequential I/O, fast):
      For each video -> read frames sequentially -> resize -> JPEG encode -> append

    Access phase (random, fast):
      Decompress a single JPEG from bytes -> numpy array (~0.2 ms at 256x256)

    Memory: ~20-50 KB per frame at quality 95 vs ~196 KB raw for 256x256x3.
    100 k frames approx 2-5 GB compressed vs ~19.6 GB raw.
    """

    def __init__(self):
        self._bufs: list[bytes] = []      # JPEG bytes per frame
        self._net_wh: tuple[int, int] = (0, 0)

    @property
    def net_wh(self) -> tuple[int, int]:
        return self._net_wh

    def __len__(self) -> int:
        return len(self._bufs)

    def add_video(self, video_path: str, net_wh: tuple[int, int],
                  stride: int = 1) -> tuple[int, int, int, int, int]:
        """Sequentially decode a video into the cache.

        Returns (start_idx, n_frames, n_total_raw, orig_w, orig_h).
        """
        self._net_wh = net_wh
        net_w, net_h = net_wh
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_total = _actual_frame_count(cap, n_reported)

        indices = list(range(0, n_total, stride))
        start_idx = len(self._bufs)
        n_frames = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pbar = tqdm(indices, desc=f"    cache {Path(video_path).name}",
                    unit="fr", dynamic_ncols=True, leave=False)
        read_pos = 0
        for orig_idx in pbar:
            if read_pos != orig_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                read_pos = orig_idx
            ret, bgr = cap.read()
            if not ret:
                break
            read_pos = orig_idx + 1
            small = cv2.resize(bgr, (net_w, net_h),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, encode_params)
            if ok:
                self._bufs.append(buf.tobytes())
                n_frames += 1
        cap.release()
        return start_idx, n_frames, n_total, orig_w, orig_h

    def get_frame(self, idx: int) -> np.ndarray:
        """Decompress frame at idx -> RGB uint8 array (H, W, 3)."""
        buf = np.frombuffer(self._bufs[idx], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def memory_mb(self) -> float:
        return sum(len(b) for b in self._bufs) / (1024 * 1024)


# --- Network ----------------------------------------------------------------

class _CBR(nn.Sequential):
    """Conv 3x3 -> BatchNorm -> LeakyReLU."""
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
        self.e0 = _CBR(6,    c[0])
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
        x  = torch.cat([prev, curr], dim=1)
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b  = self.bot(e3)

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:],
                                 mode="bilinear", align_corners=False)

        d2 = self.d2(torch.cat([up(b,  e2), e2], 1))
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


# --- Dataset ----------------------------------------------------------------

_EMPTY_ANN = np.empty((0, 5), dtype=np.float32)


class FlowPairDataset(Dataset):
    """
    Yields consecutive frame pairs from the JPEG cache.

    Each sample = (prev_rgb, curr_rgb, annotations, orig_hw).
    Frames are decompressed from the cache on access (~0.2 ms each).
    
    If indices is provided, only yield pairs at those indices (for train/val split).
    """

    def __init__(self, cache: FrameCache,
                 video_infos: list[dict],
                 augment: bool = False,
                 indices: list[int] | None = None):
        self.cache   = cache
        self.augment = augment
        # Build flat list of (cache_idx_prev, cache_idx_curr, orig_frame_idx, orig_h, orig_w, csv_data_ref)
        all_pairs: list[tuple[int, int, int, int, int, dict]] = []

        for info in video_infos:
            start = info["start"]
            n     = info["n_frames"]
            csv_d = info["csv"]
            oh    = info["orig_h"]
            ow    = info["orig_w"]
            stride = info["stride"]

            for i in range(n - 1):
                orig_idx_curr = (start - info["start_raw_offset"]) * stride + (i + 1) * stride
                # simplified: original frame index of the (i+1)th cached frame
                orig_idx_curr = info["orig_indices"][i + 1]
                all_pairs.append((start + i, start + i + 1,
                                   orig_idx_curr, oh, ow, csv_d))

        # If indices specified, filter to those pairs only
        if indices is not None:
            self.pairs = [all_pairs[i] for i in indices if i < len(all_pairs)]
        else:
            self.pairs = all_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ci_prev, ci_curr, orig_idx, oh, ow, csv_d = self.pairs[idx]

        prev_np = self.cache.get_frame(ci_prev)
        curr_np = self.cache.get_frame(ci_curr)

        if self.augment:
            prev_np, curr_np = _augment(prev_np, curr_np)

        prev_t = torch.from_numpy(prev_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        curr_t = torch.from_numpy(curr_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        ann = csv_d.get(orig_idx, _EMPTY_ANN)
        return prev_t, curr_t, ann, (oh, ow)


def _augment(prev: np.ndarray, curr: np.ndarray):
    if random.random() < 0.5:
        prev = np.ascontiguousarray(prev[:, ::-1])
        curr = np.ascontiguousarray(curr[:, ::-1])
    if random.random() < 0.6:
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-20, 20)
        prev = np.clip(prev.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        curr = np.clip(curr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        sigma = random.uniform(2, 8)
        noise = np.random.randn(*prev.shape).astype(np.float32) * sigma
        prev = np.clip(prev.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        curr = np.clip(curr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return prev, curr


def _collate(batch):
    prevs, currs, anns, orig_hws = zip(*batch)
    # Return the full list so loss functions can apply per-sample scaling when
    # multiple videos with different original resolutions are in one batch.
    return torch.stack(prevs), torch.stack(currs), list(anns), list(orig_hws)


# --- Loss functions ---------------------------------------------------------

def flow_vector_loss(flow_batch, anns_batch, orig_hws):
    """Per-point flow supervision from CSV vectors.

    orig_hws: list of (orig_h, orig_w) tuples, one per batch item.
    Using per-sample scale factors is critical when videos with different
    original resolutions are mixed in the same batch.
    """
    device = flow_batch.device
    B, _, H, W = flow_batch.shape

    total = torch.zeros(1, device=device)
    n_pts = 0
    for b in range(B):
        orig_h, orig_w = orig_hws[b]
        sx, sy = W / orig_w, H / orig_h
        anns = anns_batch[b]
        if len(anns) == 0:
            continue
        nx = (anns[:, 0] * sx).astype(np.intp)
        ny = (anns[:, 1] * sy).astype(np.intp)
        valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        if not valid.any():
            continue
        nx, ny = nx[valid], ny[valid]
        target = torch.tensor(
            np.stack([anns[valid, 2] * sx, anns[valid, 3] * sy], axis=1),
            device=device, dtype=torch.float32)
        flow = flow_batch[b]
        pred = torch.stack([flow[0, ny, nx], flow[1, ny, nx]], dim=1)
        total += F.smooth_l1_loss(pred, target, beta=2.0, reduction="sum")
        n_pts += len(nx)
    return total / max(n_pts, 1)


def background_loss(flow_batch, anns_batch, orig_hws, radius=3):
    """Suppress flow toward zero except near annotated points.

    orig_hws: list of (orig_h, orig_w) tuples, one per batch item.
    """
    device = flow_batch.device
    B, _, H, W = flow_batch.shape
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)

    total = torch.zeros(1, device=device)
    for b in range(B):
        orig_h, orig_w = orig_hws[b]
        sx, sy = W / orig_w, H / orig_h
        mag = flow_batch[b].norm(dim=0)
        anns = anns_batch[b]
        if len(anns) > 0:
            nx = np.clip((anns[:, 0] * sx).astype(np.intp), 0, W - 1)
            ny = np.clip((anns[:, 1] * sy).astype(np.intp), 0, H - 1)
            mask_np = np.zeros((H, W), dtype=np.uint8)
            mask_np[ny, nx] = 1
            mask_np = cv2.dilate(mask_np, kernel)
            motion_mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)
        else:
            motion_mask = torch.zeros(H, W, device=device)
        bg_mask = 1.0 - motion_mask
        n_bg = bg_mask.sum()
        if n_bg > 0:
            total += mag.mul(bg_mask).sum() / n_bg
    return total / B


_BASE_GRID_CACHE: dict[tuple, torch.Tensor] = {}

def _get_base_grid(H, W, device):
    key = (H, W, device)
    if key not in _BASE_GRID_CACHE:
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device), indexing="ij")
        _BASE_GRID_CACHE[key] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return _BASE_GRID_CACHE[key]


def photometric_loss(prev, curr, flow):
    """Warp prev with flow, compare to curr (L1)."""
    B, _, H, W = prev.shape
    grid = _get_base_grid(H, W, prev.device).expand(B, -1, -1, -1)
    grid = grid + torch.stack([flow[:, 0] / (W / 2.0),
                               flow[:, 1] / (H / 2.0)], dim=-1)
    warped = F.grid_sample(prev, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return F.l1_loss(warped, curr)


def tv_loss(flow):
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return dx + dy


# --- ONNX export -------------------------------------------------------------

class _FlowNetONNX(nn.Module):
    def __init__(self, net: FlowNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, :3], x[:, 3:])


def export_onnx(ckpt_path: str):
    p = Path(ckpt_path)
    if not p.is_file():
        sys.exit(f"Checkpoint not found: {p}")

    ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
    net_wh = tuple(ckpt["net_wh"])
    mf = ckpt.get("max_flow", MAX_FLOW)

    net = FlowNet(max_flow=mf)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    wrapper = _FlowNetONNX(net)
    wrapper.eval()

    net_w, net_h = net_wh
    dummy = torch.randn(1, 6, net_h, net_w)
    out = p.with_suffix(".onnx")

    torch.onnx.export(
        wrapper, dummy, str(out),
        input_names=["input"], output_names=["flow"],
        dynamic_axes={"input": {0: "batch"}, "flow": {0: "batch"}},
        opset_version=17,
    )
    orig_wh = ckpt.get("orig_wh", ("?", "?"))
    print(f"Exported : {out}")
    print(f"  Input  : (B, 6, {net_h}, {net_w})  [prev_rgb, curr_rgb] in [0,1]")
    print(f"  Output : (B, 2, {net_h}, {net_w})  [dx, dy] in [+/-{mf}]")
    print(f"Trained on {orig_wh[0]}x{orig_wh[1]}, "
          f"epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")


# --- Training ---------------------------------------------------------------

def train(data_dir: str, csv_path: str | None = None, val_split: float = VAL_SPLIT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    root = Path(data_dir)
    
    # Load video filter from CSV if provided
    video_filter = None
    if csv_path:
        video_filter = load_video_list(csv_path)
    
    pairs = _collect_pairs(root, video_filter=video_filter)

    # Determine net size from first video
    cap = cv2.VideoCapture(str(pairs[0][0]))
    ow0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    net_wh = _net_size(ow0, oh0)
    print(f"Net size : {net_wh[0]}x{net_wh[1]} (from {ow0}x{oh0})")

    # ── Build JPEG frame cache ──────────────────────────────────────────
    print(f"\nCaching {len(pairs)} video(s) as JPEG-in-RAM (q={JPEG_QUALITY}) ...")
    cache = FrameCache()
    video_infos: list[dict] = []

    for vid_path, csv_data in pairs:
        start, n_frames, n_raw, ow, oh = cache.add_video(
            str(vid_path), net_wh, stride=STRIDE)
        orig_indices = list(range(0, n_raw, STRIDE))[:n_frames]
        video_infos.append({
            "start": start,
            "n_frames": n_frames,
            "csv": csv_data,
            "orig_h": oh, "orig_w": ow,
            "stride": STRIDE,
            "orig_indices": orig_indices,
            "start_raw_offset": 0,
        })
        total_vecs = sum(len(v) for v in csv_data.values())
        ann_frames = len(csv_data)
        print(f"  {vid_path.name}: {n_frames} frames cached, "
              f"{total_vecs:,} flow vectors across {ann_frames} annotated frames")

    total_frames = sum(vi["n_frames"] for vi in video_infos)
    total_pairs  = sum(vi["n_frames"] - 1 for vi in video_infos)
    print(f"\nTotal    : {total_frames:,} frames ({cache.memory_mb():.0f} MB in RAM)")
    print(f"Pairs    : {total_pairs:,}")

    # ── Dataset / loader ────────────────────────────────────────────────
    # Create full dataset first to split it
    ds_full = FlowPairDataset(cache, video_infos, augment=False)
    total_pairs_full = len(ds_full)
    
    # Split into train and validation indices
    n_val = max(1, int(total_pairs_full * val_split))
    n_train = total_pairs_full - n_val
    
    indices = list(range(total_pairs_full))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"Train/Val split: {n_train} / {n_val}")
    
    # Create train and val datasets
    ds_train = FlowPairDataset(cache, video_infos, augment=True, indices=train_indices)
    ds_val = FlowPairDataset(cache, video_infos, augment=False, indices=val_indices)
    
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=_collate,
                              drop_last=False, pin_memory=False)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=_collate,
                            drop_last=False, pin_memory=False)

    # ── Model ───────────────────────────────────────────────────────────
    net = FlowNet(max_flow=MAX_FLOW).to(device)
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Params   : {n_par:,}")

    start_epoch = 1
    if RESUME_PATH and Path(RESUME_PATH).is_file():
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  (loss={ckpt['loss']:.4f})")

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    # Linear warm-up for WARMUP_EPOCHS, then cosine decay to 1% of LR.
    # Warm-up prevents large early gradient updates that derail flow learning.
    _warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=LR * 0.01)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[_warmup, _cosine], milestones=[WARMUP_EPOCHS])
    for _ in range(start_epoch - 1):
        sched.step()

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"\nTraining for {EPOCHS} epochs "
          f"(w_photo={W_PHOTO}, w_flow={W_FLOW}, "
          f"w_bg={W_BG}, w_tv={W_TV})"
          f"{' [AMP fp16]' if use_amp else ''}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        # === TRAINING ===
        net.train()
        totals = dict(loss=0.0, photo=0.0, flow_v=0.0, bg=0.0, tv=0.0)
        nb = 0

        pbar = tqdm(loader_train, desc=f"Epoch {epoch:3d}/{EPOCHS}",
                    unit="batch", dynamic_ncols=True, leave=True)
        for prev, curr, anns, orig_hws in pbar:
            prev = prev.to(device, non_blocking=True)
            curr = curr.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                flow, flow_d1, flow_d2 = net(prev, curr)
                l_photo = photometric_loss(prev, curr, flow)
                # Multi-scale flow supervision: coarser levels give the
                # encoder stronger gradient signal early in training.
                l_flow  = (flow_vector_loss(flow, anns, orig_hws)
                           + 0.5 * flow_vector_loss(flow_d1, anns, orig_hws)
                           + 0.25 * flow_vector_loss(flow_d2, anns, orig_hws))
                l_bg    = background_loss(flow, anns, orig_hws)
                l_tv    = tv_loss(flow)
                loss = (W_PHOTO * l_photo + W_FLOW * l_flow
                        + W_BG * l_bg + W_TV * l_tv)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            totals["loss"]   += loss.item()
            totals["photo"]  += l_photo.item()
            totals["flow_v"] += l_flow.item()
            totals["bg"]     += l_bg.item()
            totals["tv"]     += l_tv.item()
            nb += 1

            pbar.set_postfix(loss=f"{totals['loss']/nb:.4f}")

        # === VALIDATION ===
        net.eval()
        val_totals = dict(loss=0.0, photo=0.0, flow_v=0.0, bg=0.0, tv=0.0)
        n_val_batches = 0
        
        with torch.no_grad():
            for prev, curr, anns, orig_hws in loader_val:
                prev = prev.to(device, non_blocking=True)
                curr = curr.to(device, non_blocking=True)
                
                with torch.amp.autocast(device.type, enabled=use_amp):
                    flow = net(prev, curr)
                    if isinstance(flow, tuple):
                        flow = flow[0]  # Use only full resolution output
                    l_photo = photometric_loss(prev, curr, flow)
                    l_flow  = flow_vector_loss(flow, anns, orig_hws)
                    l_bg    = background_loss(flow, anns, orig_hws)
                    l_tv    = tv_loss(flow)
                    loss = (W_PHOTO * l_photo + W_FLOW * l_flow
                            + W_BG * l_bg + W_TV * l_tv)
                
                val_totals["loss"]   += loss.item()
                val_totals["photo"]  += l_photo.item()
                val_totals["flow_v"] += l_flow.item()
                val_totals["bg"]     += l_bg.item()
                val_totals["tv"]     += l_tv.item()
                n_val_batches += 1

        sched.step()
        # Free fragmented GPU memory and short-lived Python objects
        # between epochs to prevent OOM / segfaults in long runs.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        avg = {k: v / max(nb, 1) for k, v in totals.items()}
        avg_val = {k: v / max(n_val_batches, 1) for k, v in val_totals.items()}
        lr  = sched.get_last_lr()[0]
        print(f"  [Train] loss={avg['loss']:.4f}  photo={avg['photo']:.4f}  "
              f"flow={avg['flow_v']:.4f}  bg={avg['bg']:.4f}  "
              f"tv={avg['tv']:.4f}")
        print(f"  [Val]   loss={avg_val['loss']:.4f}  photo={avg_val['photo']:.4f}  "
              f"flow={avg_val['flow_v']:.4f}  bg={avg_val['bg']:.4f}  "
              f"tv={avg_val['tv']:.4f}  lr={lr:.2e}")

        ckpt = {
            "epoch":      epoch,
            "state_dict": net.state_dict(),
            "net_wh":     net_wh,
            "orig_wh":    (ow0, oh0),
            "max_flow":   MAX_FLOW,
            "loss":       avg["loss"],
            "val_loss":   avg_val["loss"],
        }
        if epoch % 5 == 0 or epoch == EPOCHS:
            p = save_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, p)
            print(f"  saved to {p}")
        if avg_val["loss"] < best_loss:
            best_loss = avg_val["loss"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  saved to best.pt  (val_loss={best_loss:.4f})")

    print(f"\nTraining complete.  Best loss: {best_loss:.4f}")


# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python train_flow_NEW.py <data_folder> [--csv <csv_file>] [--val-split <fraction>] [--export]")
        sys.exit(0)

    data_dir = sys.argv[1]
    csv_path = None
    val_split = VAL_SPLIT
    
    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--csv" and i + 1 < len(sys.argv):
            csv_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--val-split" and i + 1 < len(sys.argv):
            try:
                val_split = float(sys.argv[i + 1])
                val_split = max(0.0, min(1.0, val_split))  # Clamp to [0, 1]
            except ValueError:
                print(f"Warning: Invalid val-split value '{sys.argv[i + 1]}', using default {VAL_SPLIT}")
            i += 2
        elif sys.argv[i] == "--export":
            ckpt = str(Path(SAVE_DIR) / "best.pt")
            export_onnx(ckpt)
            sys.exit(0)
        else:
            i += 1

    train(data_dir=data_dir, csv_path=csv_path, val_split=val_split)
