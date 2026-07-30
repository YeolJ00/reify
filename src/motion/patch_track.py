"""Track the subject by matching its own appearance, and report how well it still matches.

Point tracking failed silently on these clips: CoTracker reported high visibility while
its points sat on the background, so a duck that visibly fell to the table measured 6 px
of motion. Visibility cannot detect that, because the points really are visible -- just
not on the object.

Normalised cross-correlation gives two numbers per frame instead of one, and the second
is the missing check:

    displacement   where the patch best matches now, relative to frame 0
    ncc peak       HOW WELL it matches there, i.e. does the object still look like itself

A generated clip can fail in two distinct ways, and only both numbers together separate
them:

    moves, peak stays high   usable rigid motion -- what a probe needs
    still,  peak stays high   genuinely static; no information, but the asset is intact
    peak collapses            the object stopped being the object (our ceramic vase
                              falls and then disintegrates into two spheres). There is no
                              rigid-body parameter that explains this at any theta, so it
                              is not a measurement problem but an asset-integrity one.

Template matching is drift-prone if chained frame to frame, so the reference patch stays
FIXED at frame 0: displacement is always measured against the original appearance, which
is what makes a falling peak meaningful rather than an accumulated error.
"""
import cv2
import numpy as np


def _parabola(a, b, c):
    """Offset of a parabola's vertex through three equally spaced samples."""
    den = a - 2.0 * b + c
    if abs(den) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))


def _subpixel(res, x, y):
    h, w = res.shape
    ox = _parabola(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    oy = _parabola(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return ox, oy


def track_patch(frames, u, v, half, search=None, ref_frame=0):
    """Match a patch from `ref_frame` in every frame by NCC.

    frames: (T, H, W, 3) uint8. (u, v): patch centre in the reference frame.
    half:   half-size of the patch in pixels.
    search: half-size of the search window around the previous best match; None
            searches the whole frame, which is slower but cannot lose a fast object.

    Returns dict of arrays: u, v (best-match centre per frame), ncc (peak value),
    plus displacement relative to the reference.
    """
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    u = int(round(u)); v = int(round(v))
    half = int(max(half, 6))
    x0, x1 = max(u - half, 0), min(u + half, W)
    y0, y1 = max(v - half, 0), min(v + half, H)
    tpl = g[ref_frame][y0:y1, x0:x1]
    if tpl.shape[0] < 6 or tpl.shape[1] < 6:
        return None

    cu = np.full(T, np.nan); cv_ = np.full(T, np.nan); peak = np.full(T, np.nan)
    last = (u, v)
    for t in range(T):
        img = g[t]
        if search is None:
            sx0, sy0 = 0, 0
            sub = img
        else:
            sx0 = int(np.clip(last[0] - search, 0, W - 1))
            sy0 = int(np.clip(last[1] - search, 0, H - 1))
            sx1 = int(np.clip(last[0] + search, 1, W))
            sy1 = int(np.clip(last[1] + search, 1, H))
            sub = img[sy0:sy1, sx0:sx1]
        if sub.shape[0] < tpl.shape[0] or sub.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, ml = cv2.minMaxLoc(res)
        # SUBPIXEL: matchTemplate peaks on the integer grid, but every parameter here is
        # a velocity fitted over ~5 frames, so +-0.5px of quantisation is the same size
        # as the real tracking noise. A parabola through the peak and its two neighbours
        # recovers the fractional offset in each axis.
        ox, oy = _subpixel(res, ml[0], ml[1])
        bx = sx0 + ml[0] + ox + tpl.shape[1] / 2.0
        by = sy0 + ml[1] + oy + tpl.shape[0] / 2.0
        cu[t] = bx; cv_[t] = by; peak[t] = mx
        last = (bx, by)

    ok = np.isfinite(cu)
    if ok.sum() < 3:
        return None
    P = np.stack([cu, cv_], 1)
    ref = P[ref_frame] if np.isfinite(P[ref_frame, 0]) else P[ok][0]
    disp = np.full(T, np.nan)
    disp[ok] = np.hypot(*(P[ok] - ref).T)
    return {"u": cu, "v": cv_, "ncc": peak, "disp": disp,
            "net_px": float(np.nanmax(disp)),
            "end_px": float(disp[ok][-1]),
            "ncc_median": float(np.nanmedian(peak)),
            "ncc_min": float(np.nanmin(peak)),
            "ncc_end": float(np.nanmedian(peak[ok][-5:]))}
