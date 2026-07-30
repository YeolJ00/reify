"""Physical observables with error bars, read straight off a track.

This replaces the hand-weighted "motion signature" machinery. The three probes we
stage are each directly invertible, so there is nothing to search:

    drop     restitution e = |v_up| / |v_down|          (dimensionless)
    slide    friction    mu = |a| / g                   (needs the px<->m scale)
    collide  mass ratio  m_target/m_mover
                         = (v_mover_pre - v_mover_post) / v_target_post

Two of the three need no calibration at all, because they are ratios of speeds
measured in the same units in the same clip.

Everything is a velocity, and every velocity comes from a least-squares fit over
a few frames, which yields a standard error for free. That single fact removes
most of the previous complexity:

  * No hand-tuned feature weights, and therefore no need to re-score under
    several objectives to measure how arbitrary the weights were.
  * No "did it move" threshold in object-widths. The precondition becomes a
    SIGNIFICANCE TEST: if the fitted approach speed is not distinguishable from
    zero, the motion did not happen. One principle, no magic numbers.
  * "No bounce" stops being a rejected take and becomes a real measurement,
    e = 0 +- sigma, which is what it always was.

Uncertainty is propagated analytically, so each parameter arrives as
value +- interval and an interval that spans the plausible range IS the
"not identifiable" verdict -- no separate vocabulary needed.
"""
import numpy as np

G = 9.81

# The physically possible range for each parameter. An estimate outside it is not
# an imprecise measurement, it is a take that CONTRADICTS the model -- a mover
# that gains speed through an impact implies a negative mass, and a rebound
# faster than the fall implies energy created at the contact. Such takes are
# reported and then excluded from the combination, because averaging an
# impossible number with a possible one produces a meaningless middle.
ADMISSIBLE = {"restitution": (0.0, 1.0), "friction": (0.0, 2.0),
              "mass_ratio": (1e-3, np.inf)}


def admissible(kind, value):
    lo, hi = ADMISSIBLE[kind]
    return bool(np.isfinite(value) and lo <= value <= hi)
SIGMA_FLOOR_PX = 1.5      # tracker noise floor; residuals alone can look too good
SIG_K = 3.0               # a velocity counts as real at 3 standard errors
WIN = 5                   # frames either side of an event used to fit a velocity


def polyfit_se(t, x, deg, sigma_floor=SIGMA_FLOOR_PX):
    """Least-squares polynomial with standard errors on the coefficients.

    Returns (coeffs_low_to_high, standard_errors). The residual variance is
    floored at sigma_floor**2: a smoothed or interpolated track can have tiny
    residuals while still being wrong by a pixel or two, and we must not report
    an error bar smaller than the tracker can actually deliver.
    """
    t = np.asarray(t, float); x = np.asarray(x, float)
    n = len(t)
    if n < deg + 2:
        return None, None
    V = np.vander(t, deg + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(V, x, rcond=None)
    resid = x - V @ beta
    dof = max(n - (deg + 1), 1)
    s2 = max(float(resid @ resid) / dof, float(sigma_floor) ** 2)
    try:
        cov = s2 * np.linalg.inv(V.T @ V)
    except np.linalg.LinAlgError:
        return None, None
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _valid(xy):
    xy = np.asarray(xy, float)
    ok = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])
    return xy, ok


def _direction(xy, ok):
    """Unit vector of the object's overall travel, used to make motion signed.

    Projecting onto one direction rather than taking arc length matters: arc
    length is a sum of absolute steps and so accumulates tracking noise, which
    is exactly how a stationary object previously acquired 100 px of 'travel'.
    A signed projection lets noise cancel.
    """
    P = xy[ok]
    if len(P) < 2:
        return np.array([1.0, 0.0])
    d = P[-1] - P[0]
    n = float(np.hypot(*d))
    if n < 1e-6:
        # no net motion: fall back to the largest excursion so we still measure
        # something rather than dividing by zero
        j = int(np.argmax(np.hypot(*(P - P[0]).T)))
        d = P[j] - P[0]; n = float(np.hypot(*d))
    return d / n if n > 1e-6 else np.array([1.0, 0.0])


def _ratio_se(a, sa, b, sb):
    """Standard error of a/b by first-order propagation."""
    if abs(b) < 1e-9:
        return np.nan, np.nan
    r = a / b
    return r, abs(r) * np.hypot(sa / a if abs(a) > 1e-9 else np.inf, sb / b)


def velocity(xy, ok, lo, hi, direction, sigma_floor=SIGMA_FLOOR_PX):
    """Signed speed along `direction` over frames [lo, hi], with its std error."""
    idx = np.arange(len(xy))
    sel = ok & (idx >= lo) & (idx <= hi)
    if sel.sum() < 3:
        return np.nan, np.nan
    s = (xy[sel] - xy[sel][0]) @ direction
    beta, se = polyfit_se(idx[sel], s, 1, sigma_floor)
    if beta is None:
        return np.nan, np.nan
    return float(beta[1]), float(se[1])


def significant(v, se, k=SIG_K):
    return bool(np.isfinite(v) and np.isfinite(se) and se > 0 and abs(v) > k * se)


# --------------------------------------------------------------------------
# drop -> restitution
# --------------------------------------------------------------------------

def drop_observables(cen, win=WIN):
    """|v_up|/|v_down| across the landing. Image y is positive DOWNWARD."""
    xy, ok = _valid(cen)
    if ok.sum() < 8:
        return {"ok": False, "why": "too few tracked frames"}
    y = np.where(ok, xy[:, 1], np.nan)
    n = len(y)
    hit = int(np.nanargmax(y[: max(int(0.85 * n), 6)]))    # lowest point = landing
    if hit < 3 or hit > n - 4:
        return {"ok": False, "why": "no landing inside the clip"}
    down = np.array([0.0, 1.0])
    v_pre, s_pre = velocity(xy, ok, hit - win, hit, down)
    v_post, s_post = velocity(xy, ok, hit, hit + win, down)
    if not significant(v_pre, s_pre):
        return {"ok": False, "why": f"no fall: v={v_pre:.2f}+-{s_pre:.2f} px/frame",
                "v_pre": v_pre, "v_pre_se": s_pre}
    # v_post is negative when it rebounds; a non-significant v_post is a genuine
    # measurement of "did not bounce", not a failure
    if v_post > 0:
        # still descending after the deepest point: no rebound occurred at all.
        # e = 0 is the measurement, and its error bar is what the noise allows.
        e, se = 0.0, float(s_post / abs(v_pre))
    else:
        e = abs(v_post) / abs(v_pre)
        se = abs(e) * np.hypot(s_post / v_post, s_pre / v_pre)
    return {"ok": True, "value": float(e), "se": float(se), "hit": hit,
            "kind": "restitution",
            "v_pre": v_pre, "v_pre_se": s_pre, "v_post": v_post, "v_post_se": s_post,
            # distinguishes "measured a bounce" from "measured no bounce": both are
            # results, but only the first constrains restitution away from zero
            "bounced": bool(significant(v_post, s_post) and v_post < 0)}


# --------------------------------------------------------------------------
# slide -> friction
# --------------------------------------------------------------------------

def slide_observables(cen, px_per_m, fps, win=None):
    """Coulomb friction from the constant deceleration of a sliding object.

    mu = |a| / g, with a converted from px/frame^2 into m/s^2. This is the only
    observable of the three that needs the pixel<->metre scale, and therefore the
    only one carrying that systematic.
    """
    xy, ok = _valid(cen)
    if ok.sum() < 8:
        return {"ok": False, "why": "too few tracked frames"}
    d = _direction(xy, ok)
    idx = np.arange(len(xy))
    s = np.where(ok, (xy - xy[ok][0]) @ d, np.nan)
    # fit only while it is still going: past the turning point a quadratic would
    # be extrapolating a reversal the object never performed
    stop = int(np.nanargmax(s))
    stop = max(stop, 6)
    sel = ok & (idx <= stop)
    if sel.sum() < 5:
        return {"ok": False, "why": "too short a run to fit a deceleration"}
    beta, se = polyfit_se(idx[sel], s[sel], 2)
    if beta is None:
        return {"ok": False, "why": "deceleration fit failed"}
    v0, sv0 = float(beta[1]), float(se[1])
    a, sa = 2.0 * float(beta[2]), 2.0 * float(se[2])
    if not significant(v0, sv0):
        return {"ok": False, "why": f"never moved: v0={v0:.2f}+-{sv0:.2f} px/frame",
                "v0": v0, "v0_se": sv0}
    k = fps ** 2 / (px_per_m * G)
    return {"ok": True, "value": float(abs(a) * k), "se": float(sa * k),
            "kind": "friction",
            "v0": v0, "v0_se": sv0, "a_px": a, "a_px_se": sa, "stop_frame": stop,
            "decelerating": bool(a < 0)}


# --------------------------------------------------------------------------
# collide -> mass ratio
# --------------------------------------------------------------------------

def collide_observables(mover, target, win=WIN):
    """m_target/m_mover from momentum conservation across the impact.

    m_m v_pre = m_m v_post + m_t v_t   ->   m_t/m_m = (v_pre - v_post)/v_t
    Dimensionless: all three speeds are in px/frame in the same clip, so the
    pixel scale and the frame rate cancel exactly.
    """
    mv, mok = _valid(mover); tg, tok = _valid(target)
    if mok.sum() < 8 or tok.sum() < 8:
        return {"ok": False, "why": "too few tracked frames"}
    d = _direction(mv, mok)
    idx = np.arange(len(mv))
    st = np.where(tok, (tg - tg[tok][0]) @ d, np.nan)

    # impact = first frame at which the target's displacement is significant
    base = np.nanstd(st[:max(int(0.15 * len(st)), 3)])
    thr = max(SIG_K * max(base, SIGMA_FLOOR_PX), SIG_K * SIGMA_FLOOR_PX)
    moved = np.where(np.abs(st) > thr)[0]
    if not len(moved):
        return {"ok": False, "why": f"target never moved (max {np.nanmax(np.abs(st)):.1f}px "
                                    f"< {thr:.1f}px)"}
    hit = int(moved[0])
    if hit < 3 or hit > len(mv) - 4:
        return {"ok": False, "why": f"impact at frame {hit} is too close to the clip edge"}

    v_pre, s_pre = velocity(mv, mok, hit - win, hit, d)
    v_post, s_post = velocity(mv, mok, hit, hit + win, d)
    v_t, s_t = velocity(tg, tok, hit, hit + win, d)
    if not significant(v_pre, s_pre):
        return {"ok": False, "why": f"mover was not approaching: "
                                    f"v={v_pre:.2f}+-{s_pre:.2f} px/frame",
                "v_pre": v_pre, "v_pre_se": s_pre}
    if not significant(v_t, s_t):
        return {"ok": False, "why": f"target did not depart: v={v_t:.2f}+-{s_t:.2f} px/frame",
                "v_t": v_t, "v_t_se": s_t}
    dv = v_pre - v_post
    s_dv = float(np.hypot(s_pre, s_post))
    ratio, se = _ratio_se(dv, s_dv, v_t, s_t)
    return {"ok": True, "value": float(ratio), "se": float(se), "hit": hit,
            "kind": "mass_ratio",
            "v_pre": v_pre, "v_pre_se": s_pre, "v_post": v_post, "v_post_se": s_post,
            "v_t": v_t, "v_t_se": s_t,
            # the mover cannot speed up through an impact; report rather than reject
            "mover_sped_up": bool(dv < 0)}


# --------------------------------------------------------------------------
# combining repeated takes
# --------------------------------------------------------------------------

def combine(values, ses, spread_k=1.0):
    """Inverse-variance weighted mean, with the interval widened by real scatter.

    Two things can be wrong with a set of repeats: each may be imprecise (the
    within-take error bars), or they may disagree with each other (between-take
    scatter, which for generated video is the dominant term because the model is
    not repeatable). The honest interval is the LARGER of the two, so a set of
    tight-but-contradictory takes cannot masquerade as a precise answer.
    """
    v = np.asarray(values, float); s = np.asarray(ses, float)
    m = np.isfinite(v) & np.isfinite(s) & (s > 0)
    v, s = v[m], s[m]
    if len(v) == 0:
        return None
    w = 1.0 / s ** 2
    mean = float((w * v).sum() / w.sum())
    within = float(np.sqrt(1.0 / w.sum()))
    between = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"value": mean, "interval": float(max(within, spread_k * between)),
            "within": within, "between": between, "n": int(len(v)),
            # with a single take there is no repeatability term at all, so the
            # within-take error bar UNDERSTATES the uncertainty and must not be
            # presented as a tight interval
            "single_take": bool(len(v) == 1),
            "samples": [float(x) for x in v]}
