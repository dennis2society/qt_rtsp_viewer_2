#!/usr/bin/env python3
"""
offline_flow.py — Fast offline dense optical-flow logger
========================================================
Reads a recorded video, computes dense Farneback optical flow on a
downscaled copy of each frame, and writes per-grid-point flow vectors
to a CSV file.  No UI, no bounding boxes — just raw flow data, fast.

Output CSV columns:
    frame, timestamp_sec, flow_scale, gx, gy, dx, dy, mag

  · gx, gy     — grid point position in *original* image coordinates
  · dx, dy     — flow displacement in *original* image pixels
  · mag        — flow magnitude in original-image pixels
  · flow_scale — downscale factor used for flow computation

Only vectors with magnitude ≥ --min-mag are written (default 1.0 px).

Usage:
    python3 offline_flow.py video.mp4
    python3 offline_flow.py *.mp4                       # multiple files
    python3 offline_flow.py video.mp4 --flow-scale 0.25 --grid-step 8
    python3 offline_flow.py /path/to/folder/             # batch all videos
    python3 offline_flow.py *.mp4 --workers 8            # parallelism limit
"""

import argparse
import csv as _csv
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

CSV_FIELDS = ["frame", "timestamp_sec", "flow_scale",
              "gx", "gy", "dx", "dy", "mag"]


def compute_flow(gray_prev: np.ndarray, gray_curr: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow → (H, W, 2) float32."""
    return cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def process_video(video_path: Path, flow_scale: float, grid_step: int,
                   min_mag: float, bar_pos: int = 0) -> Path:
    """Process one video file and write the companion CSV.  Returns CSV path."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = flow_scale
    proc_w = max(1, int(orig_w * scale))
    proc_h = max(1, int(orig_h * scale))
    inv_scale = 1.0 / scale  # to map back to original coords

    csv_path = video_path.with_name(video_path.stem + "_dense_flow.csv")

    grid_step = grid_step
    min_mag = min_mag

    # Pre-compute sampling grid in flow-resolution coords
    gy_coords = np.arange(grid_step // 2, proc_h, grid_step)
    gx_coords = np.arange(grid_step // 2, proc_w, grid_step)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")
    prev_gray = cv2.cvtColor(
        cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )

    rows_written = 0
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(CSV_FIELDS)

        for frame_no in tqdm(range(1, n_frames),
                             desc=f"{video_path.name}",
                             unit="fr", dynamic_ncols=True,
                             position=bar_pos, leave=True):
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(
                cv2.resize(frame, (proc_w, proc_h),
                           interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )

            flow = compute_flow(prev_gray, curr_gray)  # (H, W, 2)

            # Sample on the grid
            flow_sampled = flow[gy_coords[:, None], gx_coords[None, :]]  # (Ny, Nx, 2)
            mag_sampled = np.sqrt(
                flow_sampled[..., 0] ** 2 + flow_sampled[..., 1] ** 2
            )

            # Filter by min magnitude
            mask = mag_sampled >= (min_mag * scale)  # threshold in flow-res pixels
            if mask.any():
                ts = frame_no / fps
                iy, ix = np.nonzero(mask)
                for i in range(len(iy)):
                    y_idx, x_idx = int(iy[i]), int(ix[i])
                    gx = int(gx_coords[x_idx] * inv_scale)
                    gy = int(gy_coords[y_idx] * inv_scale)
                    dx = float(flow_sampled[y_idx, x_idx, 0]) * inv_scale
                    dy = float(flow_sampled[y_idx, x_idx, 1]) * inv_scale
                    mag = float(mag_sampled[y_idx, x_idx]) * inv_scale
                    writer.writerow([
                        frame_no, f"{ts:.4f}", scale,
                        gx, gy, f"{dx:.2f}", f"{dy:.2f}", f"{mag:.3f}",
                    ])
                    rows_written += 1

            prev_gray = curr_gray

    cap.release()
    return csv_path


def _worker(job: tuple) -> str:
    """Top-level worker function for multiprocessing."""
    video_path, flow_scale, grid_step, min_mag, bar_pos = job
    csv_path = process_video(Path(video_path), flow_scale, grid_step,
                             min_mag, bar_pos)
    return str(csv_path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="+",
                   help="Video file(s) or directory of videos")
    p.add_argument("--flow-scale", type=float, default=0.35,
                   help="Downscale factor for flow computation (default 0.35)")
    p.add_argument("--grid-step", type=int, default=8,
                   help="Grid spacing in flow-resolution pixels (default 8)")
    p.add_argument("--min-mag", type=float, default=1.0,
                   help="Min flow magnitude (original pixels) to log (default 1.0)")
    p.add_argument("--workers", type=int, default=4,
                   help="Max parallel workers (default 4)")
    args = p.parse_args()

    # Collect all video files from positional args
    videos: list[Path] = []
    for inp in args.input:
        path = Path(inp)
        if path.is_file():
            videos.append(path)
        elif path.is_dir():
            found = sorted(p for p in path.iterdir()
                           if p.suffix.lower() in VIDEO_EXTS)
            if not found:
                print(f"No video files found in: {path}", file=sys.stderr)
            videos.extend(found)
        else:
            print(f"Not a file or directory: {path}", file=sys.stderr)

    if not videos:
        sys.exit("No video files to process.")

    if len(videos) == 1:
        csv_path = process_video(videos[0], args.flow_scale, args.grid_step,
                                 args.min_mag, bar_pos=0)
        print(f"\nDone: {csv_path}")
    else:
        print(f"Processing {len(videos)} video(s) with up to "
              f"{args.workers} workers\n")
        jobs = [
            (str(v), args.flow_scale, args.grid_step, args.min_mag, i)
            for i, v in enumerate(videos)
        ]
        with mp.Pool(args.workers, maxtasksperchild=1) as pool:
            results = pool.map(_worker, jobs)
            pool.close()
            pool.join()
        # Clear tqdm lines
        print("\n" * len(videos))
        for r in results:
            print(f"Done: {r}")


if __name__ == "__main__":
    main()
