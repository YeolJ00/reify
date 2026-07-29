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


def drop_signature(y_image, x_image=None, hold_frames=3, still_eps=0.015):
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
    rebound_fraction = float(reb / drop)

    # PHYSICAL PLAUSIBILITY, the same idea already applied to collisions.
    #  * rebound height / drop height is e^2, so a 0.8 reading needs e ~ 0.89 — a
    #    superball. Ordinary props do not do that, and in practice such readings came
    #    from a tracker drifting or a tall object TOPPLING (tipping lifts the centroid,
    #    which looks exactly like a spectacular bounce).
    #  * a real bounce comes back DOWN; if the trace ends at its highest point the object
    #    was still rising when the clip ended, so nothing was measured.
    #  * a tiny fall makes the ratio noise.
    if rebound_fraction > 0.80:
        return None
    if drop < 20.0:
        return None
    i_peak = i_hit + int(np.argmax(after))
    if i_peak >= n - 2 and rebound_fraction > 0.15:
        return None
    rebound_fraction = float(np.clip(rebound_fraction, 0.0, 1.5))

    # settle: last moment the trace still moves appreciably, in units of the drop
    v = np.abs(np.diff(h)) / drop
    moving = np.where(v[i_hit:] > still_eps)[0]
    settle_frac = float((moving[-1] + 1) / max(n - i_hit, 1)) if len(moving) else 0.0

    # FRICTION CHANNEL. Restitution lives in the vertical rebound; friction lives in how
    # far the object travels sideways after it lands. Without this the signature is blind
    # to friction and no amount of joint fitting can recover it.
    slide_frac = 0.0
    if x_image is not None:
        xs = _clean(x_image)
        if xs is not None and len(xs) == n:
            slide_frac = float(np.clip(abs(xs[-1] - xs[i_hit]) / drop, 0.0, 3.0))

    return {"rebound_fraction": rebound_fraction,
            "settle_frac": settle_frac,
            "fall_frac": float(i_hit / n),
            "slide_frac": slide_frac,
            "drop_px": float(drop)}


# how much each feature matters, and the scale on which a difference is "large"
WEIGHTS = {"rebound_fraction": 1.0, "settle_frac": 0.45, "fall_frac": 0.35,
           "slide_frac": 0.8}   # the friction channel


def signature_distance(a, b):
    """Weighted distance between two drop signatures (0 = identical)."""
    if a is None or b is None:
        return 1e6
    return float(np.sqrt(sum(w * (a[k] - b[k]) ** 2 for k, w in WEIGHTS.items())
                         / sum(WEIGHTS.values())))


def describe(sig):
    if sig is None:
        return "no landing detected"
    return (f"rebound {sig['rebound_fraction']*100:.0f}%, "
            f"slides {sig.get('slide_frac',0)*100:.0f}% of the drop sideways, "
            f"settles over {sig['settle_frac']*100:.0f}%, lands {sig['fall_frac']*100:.0f}% in")


# ---------------------------------------------------------------------------
# Signatures for the other two experiments. Each is expressed in units of the
# object's own apparent SIZE, so like the drop signature it is scale-free and
# does not care where the object was or how big the image is.
# ---------------------------------------------------------------------------

def _path_len(xy):
    d = np.diff(xy, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])


def slide_signature(xy, size_px, moving_eps=0.03):
    """How a sliding/rolling object gives up its speed -> friction.

    total_travel  how far it goes, in object-widths
    decel_ratio   distance covered in the second half vs the first half
                  (1.0 = never slowed, 0.0 = stopped dead early)
    stop_frac     when it comes to rest, as a fraction of the clip
    """
    xy = np.asarray(xy, float)
    m = ~np.isnan(xy[:, 0])
    if m.sum() < 6 or size_px <= 1:
        return None
    P = xy[m]
    d = _path_len(P)
    total = float(d[-1] / size_px)
    if total < 0.15:
        return None                              # never really moved
    half = len(d) // 2
    first = d[half] - d[0]
    second = d[-1] - d[half]
    decel = float(np.clip(second / (first + 1e-6), 0.0, 2.0))
    step = np.diff(d) / max(size_px, 1e-6)
    mv = np.where(step > moving_eps)[0]
    stop_frac = float((mv[-1] + 1) / len(step)) if len(mv) else 0.0
    return {"total_travel": total, "decel_ratio": decel, "stop_frac": stop_frac}


SLIDE_WEIGHTS = {"total_travel": 0.35, "decel_ratio": 1.0, "stop_frac": 0.6}


def slide_distance(a, b):
    if a is None or b is None:
        return 1e6
    # total_travel is compared in log space: it spans a wide range and we care about
    # relative error, not absolute object-widths
    da = np.log1p(a["total_travel"]) - np.log1p(b["total_travel"])
    rest = sum(w * (a[k] - b[k]) ** 2 for k, w in SLIDE_WEIGHTS.items() if k != "total_travel")
    return float(np.sqrt((SLIDE_WEIGHTS["total_travel"] * da ** 2 + rest)
                         / sum(SLIDE_WEIGHTS.values())))


def collide_signature(mover_xy, target_xy, size_px):
    """Momentum transfer -> the mass ratio.

    transfer      how far the target is knocked, per unit of the mover's approach
    mover_kept    how much of its travel the mover keeps after the hit
    impact_frac   when the hit happens within the clip
    """
    mv = np.asarray(mover_xy, float); tg = np.asarray(target_xy, float)
    mm = ~np.isnan(mv[:, 0]); tm = ~np.isnan(tg[:, 0])
    if mm.sum() < 6 or tm.sum() < 6 or size_px <= 1:
        return None
    t0 = tg[tm][0]
    moved = np.where(tm & (np.hypot(*(np.nan_to_num(tg - t0, nan=0.0).T)) > 0.25 * size_px))[0]
    if not len(moved):
        return None                              # nothing was struck
    hit = int(moved[0])
    dm = _path_len(np.nan_to_num(mv, nan=0.0)) / size_px
    dt = _path_len(np.nan_to_num(tg, nan=0.0)) / size_px
    approach = float(dm[hit] - dm[0])
    if approach < 0.2:
        return None                              # the mover never actually came in
    transfer = float((dt[-1] - dt[hit]) / approach)
    kept = float((dm[-1] - dm[hit]) / approach)
    # PHYSICAL PLAUSIBILITY. A struck object cannot leave the impact with more motion
    # than it arrived with, and a "collision" that barely moves the target is not the
    # experiment we asked for. Either way there is no mass information in the clip.
    if kept > 1.15 or transfer < 0.05:
        return None
    return {"transfer": float(np.clip(transfer, 0.0, 4.0)),
            "mover_kept": float(np.clip(kept, 0.0, 1.15)),
            "impact_frac": float(hit / len(dm))}


# impact_frac is set by the LAUNCH VELOCITY — a per-clip nuisance we do not care about —
# not by the material. Weighting it heavily penalised timing mismatches that carry no
# physical information and pushed otherwise-good collisions over the rejection threshold
# (the apple's textbook take scored 0.404 against a 0.30 gate almost entirely on timing).
COLLIDE_WEIGHTS = {"transfer": 1.0, "mover_kept": 0.7, "impact_frac": 0.12}


def collide_distance(a, b):
    if a is None or b is None:
        return 1e6
    return float(np.sqrt(sum(w * (a[k] - b[k]) ** 2 for k, w in COLLIDE_WEIGHTS.items())
                         / sum(COLLIDE_WEIGHTS.values())))
