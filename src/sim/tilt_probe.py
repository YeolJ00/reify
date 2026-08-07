"""Tilt probe: friction from slip angle, and the design rule behind choosing it.

WHY THIS PROBE, after a long detour. Every experiment in the judge investigation used a
DROP and tested RESTITUTION. That is the worst available pairing: higher restitution means
more bounces means more pixel motion, so the parameter is confounded with motion magnitude
BY CONSTRUCTION. Measured across 240 SPSA evaluations, rho(judge score, motion magnitude)
= -0.884 while the recovered ordering had rho = 0.000 against the material prior. The judge
was being handed a parameter whose only visible signature is how much things move.

The selection rule that follows: prefer probes whose parameter signature is NOT motion
magnitude.

    parameter          probe                signature                 magnitude-confounded?
    restitution        drop                 bounce count/height       YES, structurally
    friction           tilt                 SLIP ANGLE (threshold)    no, if angle is varied
    centre of mass     tilt / topple        topple DIRECTION          no, direction not size
    stiffness (cloth)  drape                fold shape at rest        no, static
    stiffness (soft)   press-release        deformation shape         no, matched displacement

Implementation note: the incline is produced by ROTATING GRAVITY, not the ground plane.
g = (g sin(theta), 0, -g cos(theta)) is exactly equivalent for the dynamics and keeps the
existing flat-ground contact path. The render must rotate the table and camera together by
the same angle so the clip shows a tilted surface rather than objects sliding on a level
one.

MEASURED, sim-side ground truth (no judge involved):
    book       corr(mu, tan(slip_angle)) = +0.918
    brass_pot  corr(mu, tan(slip_angle)) = +0.906
    baseball   corr = +0.319   <-- a SPHERE ROLLS. It has no slip threshold and saturates
                                   near 10.5 deg for every mu. Probe choice depends on the
                                   object's geometry, not only on the parameter.

MAGNITUDE CONTROL, and this is the point of the probe. At a FIXED tilt angle, slide
distance still falls with mu, which is a motion-magnitude cue and would reproduce the old
confound. The fix is to render clips at MATCHED SLIDE DISTANCE and let the TILT ANGLE vary:
then every candidate moves the same amount, a magnitude-reader scores at chance, and only
the angle-versus-material relationship can produce a signal. slide_matched_angle() below
solves for the angle that yields a target slide distance for a given mu.
"""
import numpy as np

G = 9.81


def tilt_gravity(theta_rad, g=G):
    """Gravity for an incline of theta, with the plane left flat."""
    return (float(g * np.sin(theta_rad)), 0.0, float(-g * np.cos(theta_rad)))


def slip_angle_from_mu(mu):
    """Ideal Coulomb prediction, for checking the simulator rather than for fitting."""
    return float(np.arctan(mu))


def scan_slip(run_at_angle, angles_rad, thresh_m=0.02):
    """Smallest angle at which the body moves more than thresh_m.

    run_at_angle(theta) -> slide distance in metres, from rest, over a fixed hold time.
    Returns (slip_angle_rad, distances). None if it never slips.
    """
    d = [float(run_at_angle(t)) for t in angles_rad]
    for t, x in zip(angles_rad, d):
        if np.isfinite(x) and x > thresh_m:
            return float(t), d
    return None, d


def slide_matched_angle(run_at_angle, target_m, angles_rad, tol=0.15):
    """Angle whose slide distance is closest to target_m.

    This is what makes the probe magnitude-controlled: fix how far the object travels and
    let the angle carry the information. Returns (angle_rad, achieved_m) or (None, None)
    if no angle in the range lands within tol (relative) of the target.
    """
    best, bd = None, None
    for t in angles_rad:
        x = float(run_at_angle(t))
        if not np.isfinite(x):
            continue
        if bd is None or abs(x - target_m) < abs(bd - target_m):
            best, bd = float(t), x
    if bd is None or abs(bd - target_m) > tol * max(target_m, 1e-6):
        return None, bd
    return best, bd


# Objects whose base is flat enough that they slide rather than roll. Spheres are excluded
# on measurement, not on intuition: the baseball's slip angle is flat in mu.
SLIDES = ("book", "brass_pot", "ceramic_vase", "wooden_bowl")
ROLLS = ("baseball", "apple")
