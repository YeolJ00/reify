"""Motion signatures: fit HOW something moves, not where every pixel went.

A pixel-space loss asks the simulator to reproduce a generated clip frame by frame. That
is the wrong question for generated video: we measured that a Cosmos clip disagrees with
*any* valid physics by ~26 px, which swamps the ~10 px that the material actually
contributes. Matching pixels therefore mostly measures the video's sloppiness.

A drop's material shows up in a handful of scale-free numbers instead:

    rebound_fraction  how high it comes back, as a fraction of how far it fell
                      -> this essentially IS restitution
    settle_frac       how long after impact the motion dies away
                      -> the damping / decay envelope
    fall_frac         when the impact happens within the clip
                      -> guards against matching a clip that never landed

These are invariant to the things we neither control nor care about: a constant offset
between the tracker's centroid and the sim's body origin, a small timing shift, lateral
drift, and modest scale error. Which is the point — "how it rings, not where every pixel
goes".
"""
import numpy as np


def _clean(y, smooth=3):
    """Interpolate NaNs and lightly smooth a 1-D trace."""
    y = np.asarray(y, float)
    idx = np.arange(len(y))
    m = ~np.isnan(y)
    if m.sum() < 4:
        return None
    y = np.interp(idx, idx[m], y[m])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        y = np.convolve(y, k, mode="same")
        y[:smooth] = y[smooth]
        y[-smooth:] = y[-smooth - 1]
    return y


def drop_signature(y_image, hold_frames=3, still_eps=0.015):
    """y_image: object's image-y per frame (DOWN is positive). Returns features or None.

    Everything is computed from height h = -y, in units of the drop itself, so the
    signature does not depend on where the object was or how big the image is.
    """
    y = _clean(y_image)
    if y is None:
        return None
    h = -y
    n = len(h)
    h0 = float(np.mean(h[:hold_frames]))
    i_hit = int(np.argmin(h[: max(int(0.85 * n), 5)]))     # deepest point = the landing
    drop = h0 - float(h[i_hit])
    if drop <= 1e-6 or i_hit >= n - 3:
        return None                                        # never actually landed
    after = h[i_hit:]
    reb = float(after.max()) - float(h[i_hit])
    rebound_fraction = float(np.clip(reb / drop, 0.0, 1.5))

    # settle: last moment the trace still moves appreciably, in units of the drop
    v = np.abs(np.diff(h)) / drop
    moving = np.where(v[i_hit:] > still_eps)[0]
    settle_frac = float((moving[-1] + 1) / max(n - i_hit, 1)) if len(moving) else 0.0

    return {"rebound_fraction": rebound_fraction,
            "settle_frac": settle_frac,
            "fall_frac": float(i_hit / n),
            "drop_px": float(drop)}


# how much each feature matters, and the scale on which a difference is "large"
WEIGHTS = {"rebound_fraction": 1.0, "settle_frac": 0.45, "fall_frac": 0.35}


def signature_distance(a, b):
    """Weighted distance between two drop signatures (0 = identical)."""
    if a is None or b is None:
        return 1e6
    return float(np.sqrt(sum(w * (a[k] - b[k]) ** 2 for k, w in WEIGHTS.items())
                         / sum(WEIGHTS.values())))


def describe(sig):
    if sig is None:
        return "no landing detected"
    return (f"rebound {sig['rebound_fraction']*100:.0f}% of the drop, "
            f"settles over {sig['settle_frac']*100:.0f}% of the clip, "
            f"lands {sig['fall_frac']*100:.0f}% in")
