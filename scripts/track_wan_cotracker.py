"""Robustly track the ball in a Wan-generated video with CoTracker3 and save the
tracks for the 3D recovery. Seeds query points inside the ball's frame-0 colour
mask, then follows them through the blur/squash of impact (where blob or LK
tracking fails). Saves tracks, per-point visibility, and the visible centroid.

Run (video env): CUDA_VISIBLE_DEVICES=<g> \
  /home/jooyeolyun/anaconda3/envs/video/bin/python scripts/track_wan_cotracker.py \
      --npz outputs/i2v_wan5b_seed0.npz
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.track.cotracker import seed_in_mask, track_points  # noqa: E402


def ball_mask0(f, H):
    r, g, b = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
    m = ((r > 150) & (r < 252) & (g > 85) & (g < 200) & (b > 72) & (b < 192)
         & (r - b > 18) & (r - g > 16)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m[int(0.55 * H):] = 0                                    # ball starts in the upper scene
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/i2v_wan5b_seed0.npz")
    ap.add_argument("--n-queries", type=int, default=48)
    args = ap.parse_args()

    frames = np.load(REPO / args.npz)["frames"]
    H = frames[0].shape[0]
    q = seed_in_mask(ball_mask0(frames[0], H), n=args.n_queries, seed=0)
    print(f"seeded {len(q)} query points on the ball at frame 0")
    tracks, vis = track_points(frames, q, device="cuda")
    cen = np.array([tracks[t, vis[t]].mean(0) if vis[t].any() else [np.nan, np.nan]
                    for t in range(len(frames))])
    out = REPO / "outputs" / "wan_ball"; out.mkdir(parents=True, exist_ok=True)
    dst = out / (Path(args.npz).stem.replace("i2v_wan5b_", "cotrack_") + ".npz")
    np.savez(dst, tracks=tracks, vis=vis, cen=cen, q=q)
    print(f"tracked {vis.mean() * 100:.0f}% visible over {len(frames)} frames -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
