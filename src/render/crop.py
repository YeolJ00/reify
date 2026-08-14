"""Per-object framing from one rendered scene.

DEFAULT IS THE FULL FRAME, with the object NAMED in the prompt. Tight crops were the first
design -- render once, crop N times, keep neighbours out -- but two things argue against
them for this probe. The tilt itself is the thing the motion is being judged against, and a
tight box deletes the table edge and the incline that make a slip angle readable. And the
one capability measured reliable in this model is naming the object: open-ended questioning
gave "a rubber duck, made of rubber" and "a large, metallic pot with a lid" 4/4 correct.
Disambiguating by words is therefore cheaper and safer than disambiguating by pixels.

Cropping is kept for cases where an object is small in frame or the scene is crowded --
crop_box() still works and PAD_CONTEXT gives a wide, context-preserving box rather than a
tight one.

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
    """(N,3) world -> (N,2) pixels, via src.render.camera.Camera.

    Camera.project returns (uv, DEPTH), not (uv, valid). Unpacking the second value as a
    validity flag made every non-zero depth "valid", including NEGATIVE depths -- points
    BEHIND the camera. Those project to mirrored coordinates far outside the frame and, since
    crop_box takes the min/max over all points, a single one silently blew the box up to the
    whole frame. Validity is depth > 0.
    """
    uv, depth = cam.project(np.asarray(pts, np.float64))
    return np.asarray(uv, float), np.asarray(depth, float) > 0.0


PAD_TIGHT, PAD_CONTEXT = 0.12, 1.20
# A crop may never be smaller than this fraction of the frame in either dimension. An
# absolute pixel floor (72 px) let a barely-moving object collapse to a 74x74 thumbnail --
# 2% of the frame -- which is a texture patch, not a scene: the table, the incline and the
# object's relationship to them are all gone. Framed as a fraction, the crop stays a
# recognisable view of the scene no matter how little the object moves.
MIN_FRAC = 0.60


def crop_box(cam, pts_over_time, width, height, pad=PAD_CONTEXT, min_px=72,
             min_frac=MIN_FRAC):
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
    # floor the size: never below min_px absolute, never below min_frac of the frame
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    fw, fh = max(min_px, min_frac * width), max(min_px, min_frac * height)
    if x1 - x0 < fw:
        x0, x1 = cx - fw / 2, cx + fw / 2
    if y1 - y0 < fh:
        y0, y1 = cy - fh / 2, cy + fh / 2
    # a floored box can run off the edge; slide it back inside rather than clipping it
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= (x1 - width); x1 = width
    if y1 > height:
        y0 -= (y1 - height); y1 = height
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
