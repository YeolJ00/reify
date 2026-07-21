"""Point tracking on the rendered video (architecture step 4) and attachment of
tracks to the cloth surface via frame-0 barycentric coordinates.

Tracking is chained pyramidal Lucas-Kanade with forward-backward verification:
a track is killed the first frame LK fails or the backward check exceeds
fb_max_px (this is the step-8 'reject high-residual observations' gate at the
tracker level).
"""

import cv2
import numpy as np


def track_video(frames: np.ndarray, max_corners=300, quality=0.01, min_dist=6, fb_max_px=1.0):
    """frames (F,H,W,3) uint8 -> tracks (F,M,2), valid (F,M) bool."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=max_corners, qualityLevel=quality,
                                 minDistance=min_dist)
    if p0 is None:
        raise RuntimeError("no trackable features found in frame 0")
    p0 = p0.reshape(-1, 2).astype(np.float32)
    M = len(p0)
    F = len(frames)

    tracks = np.full((F, M, 2), np.nan, dtype=np.float64)
    valid = np.zeros((F, M), dtype=bool)
    tracks[0] = p0
    valid[0] = True

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_pts = p0.copy()
    alive = np.ones(M, dtype=bool)
    for f in range(1, F):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[f - 1], gray[f], prev_pts.reshape(-1, 1, 2),
                                              None, **lk)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray[f], gray[f - 1], nxt, None, **lk)
        nxt = nxt.reshape(-1, 2)
        fb_err = np.linalg.norm(back.reshape(-1, 2) - prev_pts, axis=1)
        ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < fb_max_px)
        alive &= ok
        tracks[f, alive] = nxt[alive]
        valid[f, alive] = True
        prev_pts = nxt
    return tracks, valid


def attach_barycentric(pts: np.ndarray, verts2d: np.ndarray, tris: np.ndarray, depth: np.ndarray):
    """For each 2D point, find the frontmost frame-0 triangle containing it.

    Returns (tri_idx (M,), bary (M,3), inside (M,) bool)."""
    M = len(pts)
    tri_idx = np.full(M, -1, dtype=np.int32)
    bary_out = np.zeros((M, 3), dtype=np.float64)

    a = verts2d[tris[:, 0]]
    b = verts2d[tris[:, 1]]
    c = verts2d[tris[:, 2]]
    tri_depth = depth[tris].mean(axis=1)

    v0 = b - a
    v1 = c - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom[np.abs(denom) < 1e-12] = np.inf

    for m in range(M):
        v2 = pts[m] - a
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        eps = -1e-6
        inside = (u >= eps) & (v >= eps) & (w >= eps)
        if inside.any():
            cands = np.flatnonzero(inside)
            k = cands[np.argmin(tri_depth[cands])]  # frontmost wins (occlusion)
            tri_idx[m] = k
            bary_out[m] = (u[k], v[k], w[k])
    return tri_idx, bary_out, tri_idx >= 0
