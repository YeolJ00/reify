"""Colour tracking of the red / blue probe balls in a generated video.

The scene makes this fiddly in two ways, so the thresholds below are calibrated from
the rendered initial frames rather than guessed:
  * the wooden TABLE is reddish-brown (r-g and r-b both large), so a naive "red" mask
    merges the ball into the tabletop — the ball is separated by being much brighter
    and much bluer (ball ~(241,170,161) vs wood ~(150,100,70));
  * the street HDRI contains a red car, so detection is restricted to the table region
    and to ball-sized, round blobs.
"""
import cv2
import numpy as np

Y_TABLE = 120          # rows above this are the street backdrop
RAD_MIN, RAD_MAX = 11.0, 52.0


def ball_masks(f):
    r, g, b = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
    red = ((r > 195) & (g > 115) & (b > 105) & (r - g > 35) & (r - b > 35)).astype(np.uint8)
    blue = ((b > 130) & (b - r > 25) & (b - g > 12)).astype(np.uint8)
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    out = []
    for m in (red, blue):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
        m[:Y_TABLE] = 0
        out.append(m)
    return out


def ball_centroid(mask, prev=None):
    """Largest round ball-sized blob; prefers the one nearest `prev` when given."""
    n, lab, st, ct = cv2.connectedComponentsWithStats(mask, 8)
    best, score = None, -1e9
    for j in range(1, n):
        a = st[j, cv2.CC_STAT_AREA]
        w, h = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        rad = np.sqrt(a / np.pi)
        if rad < RAD_MIN or rad > RAD_MAX:
            continue
        if min(w, h) / max(w, h) < 0.5 or a / (w * h + 1e-9) < 0.45:
            continue
        s = rad - (0.05 * np.hypot(*(ct[j] - prev)) if prev is not None else 0.0)
        if s > score:
            score, best = s, (ct[j].copy(), float(rad))
    return best


def track_balls(frames):
    """-> posA (F,2), radA (F,), posB (F,2), radB (F,)  with NaN where not found."""
    F = len(frames)
    A = np.full((F, 2), np.nan); B = np.full((F, 2), np.nan)
    RA = np.full(F, np.nan); RB = np.full(F, np.nan)
    pa = pb = None
    for i, f in enumerate(frames):
        mr, mb = ball_masks(f)
        ca, cb = ball_centroid(mr, pa), ball_centroid(mb, pb)
        if ca is not None:
            A[i], RA[i] = ca[0], ca[1]; pa = ca[0]
        if cb is not None:
            B[i], RB[i] = cb[0], cb[1]; pb = cb[0]
    return A, RA, B, RB
