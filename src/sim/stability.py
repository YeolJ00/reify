"""Substeps the explicit cloth solver needs to stay stable.

Split out because every script that sweeps material parameters needs it, and because
getting it wrong is silent: an unstable flag still renders, still encodes to mp4, and still
gets scored by the judge. The first J0 attempt blew a flag to a 481 m span and a 15 km tip
path and produced a perfectly good-looking video of it.

TWO conditions, not one:

    dt * sqrt(ke/mass) < ~0.3     stretch force, integrated explicitly
    kd / mass * dt     << 1       damping

Implementing only the first sent me after the wrong parameter: raising substeps for
stiffness did not stop the flag exploding, because kd=40 was the cause. Measured at 36
substeps: kd=10 stable, kd=20 unstable, while stiffness runs to ke=4e4 once its own
condition is met.
"""
import math


def substeps_for(ke, mass, fps, kd=10.0, cfl=0.15, damp_cfl=0.09, floor=32):
    """Substeps satisfying both conditions. `floor` is the config's tuned value and is a
    floor, not a starting point: dropping to what the stretch condition alone allowed blew
    up a flag that is stable at 32."""
    need_k = math.sqrt(float(ke) / float(mass)) / (cfl * float(fps))
    need_d = float(kd) / (damp_cfl * float(mass) * float(fps))
    return max(int(math.ceil(need_k)), int(math.ceil(need_d)), int(floor))


def is_exploded(traj, limit=5.0):
    """A rollout has blown up if the free corner wanders further than any real flag could.

    Cheap, and it catches the case the renderer cannot: garbage that still looks like a
    video. The flag is 1.2 m x 0.8 m, so a metres-scale excursion is unambiguous.
    """
    import numpy as np
    tip = traj[:, -1, :]
    if not np.isfinite(traj).all():
        return True, float("inf")
    span = float(np.linalg.norm(tip.max(0) - tip.min(0)))
    return span > limit, span
