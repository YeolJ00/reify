"""Robust point tracking with CoTracker3 (architecture step 4).

Unlike pyramidal LK (src/track/lk.py), CoTracker is a learned tracker that follows
query points THROUGH occlusion and deformation and reports per-frame visibility —
the failure mode that lost the ball on impact in the Wan test (M16). We seed query
points inside a frame-0 mask and follow that region; its visible centroid is the
object trajectory. (video env: torch + cotracker.)
"""
import numpy as np
import torch

_MODEL = None


def _load(device):
    global _MODEL
    if _MODEL is None:
        _MODEL = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    return _MODEL


def track_points(frames, queries_xy, query_frame=0, device="cuda"):
    """frames (F,H,W,3) uint8, queries_xy (M,2) pixel coords at query_frame.
    Returns tracks (F,M,2) float, visible (F,M) bool."""
    model = _load(device)
    vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)  # 1,F,C,H,W
    q = np.concatenate([np.full((len(queries_xy), 1), query_frame), queries_xy], 1)
    q = torch.from_numpy(q).float()[None].to(device)                             # 1,M,3 (t,x,y)
    with torch.no_grad():
        tracks, vis = model(vid, queries=q)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5


def seed_in_mask(mask, n=40, seed=0):
    """Pick ~n query points spread inside a boolean frame-0 mask (y,x -> x,y)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("empty seed mask")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)
