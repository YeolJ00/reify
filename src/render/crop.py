"""Per-object crops from one rendered scene.

The economics that motivate this: a Cycles render costs ~35 s and a judge forward ~4 s, so
rendering one clip per object is dominated by rendering. But a scene containing N objects
can be rendered ONCE and cropped N times -- the judge only needs the region containing the
object it is being asked about. Render cost becomes per-SCENE instead of per-object, which
is what makes multi-object probes affordable.

The crop box comes from the known camera and the known poses, so it costs nothing and is
exact: project the object's sphere-cover centres over the whole clip, take the union, pad.
Using the union across time rather than per-frame keeps the crop STATIC, which matters --
a box that tracks the object removes the very translation the probe is measuring.
"""
import numpy as np


def project_points(cam, pts):
    """(N,3) world -> (N,2) pixels, via src.render.camera.Camera."""
    uv, ok = cam.project(np.asarray(pts, np.float64))
    return np.asarray(uv, float), np.asarray(ok, bool)


def crop_box(cam, pts_over_time, width, height, pad=0.12, min_px=72):
    """Static box covering the object for the whole clip.

    pts_over_time: (T, K, 3) world points (sphere-cover centres are ideal).
    pad is a fraction of the box size added on every side, so the object is never flush
    against the edge -- a judge shown a clipped object reads occlusion, not physics.
    Returns (x0, y0, x1, y1) ints, clamped to the frame, or None if nothing projects.
    """
    P = np.asarray(pts_over_time, float).reshape(-1, 3)
    uv, ok = project_points(cam, P)
    uv = uv[ok]
    if len(uv) == 0:
        return None
    x0, y0 = uv[:, 0].min(), uv[:, 1].min()
    x1, y1 = uv[:, 0].max(), uv[:, 1].max()
    w, h = x1 - x0, y1 - y0
    x0 -= pad * w; x1 += pad * w
    y0 -= pad * h; y1 += pad * h
    # enforce a floor so a small object does not become a texture patch
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if x1 - x0 < min_px:
        x0, x1 = cx - min_px / 2, cx + min_px / 2
    if y1 - y0 < min_px:
        y0, y1 = cy - min_px / 2, cy + min_px / 2
    # even dimensions for the encoder, clamped to frame
    x0 = int(max(0, np.floor(x0))); y0 = int(max(0, np.floor(y0)))
    x1 = int(min(width, np.ceil(x1))); y1 = int(min(height, np.ceil(y1)))
    if (x1 - x0) % 2:
        x1 = min(width, x1 + 1) if x1 < width else x1 - 1
    if (y1 - y0) % 2:
        y1 = min(height, y1 + 1) if y1 < height else y1 - 1
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return (x0, y0, x1, y1)


def crop_clip(frames, box):
    """frames (T,H,W,3) -> cropped (T,h,w,3)."""
    x0, y0, x1, y1 = box
    return np.asarray(frames)[:, y0:y1, x0:x1]


def occupancy(frames, box):
    """Fraction of the crop that the moving content occupies, as a sanity check.

    A crop whose pixels barely change is a picture of a table, not of an object doing
    something, and scoring it tells us nothing. Returns mean |frame difference| inside the
    box, comparable to the motion-budget guard used on full frames.
    """
    c = crop_clip(frames, box).astype(np.float32)
    if len(c) < 2:
        return 0.0
    return float(np.abs(np.diff(c, axis=0)).mean())


def iou(a, b):
    """Overlap between two crop boxes. Non-zero means a judge asked about one object is
    also being shown another, which makes the answer ambiguous."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / ua)
