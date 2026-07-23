"""Recover the density RATIO from high-time-resolution observation of the impact,
via momentum conservation — bypassing the chaotic trajectory fit entirely.

Fitting the full (dense) position trajectory still fails: densely sampling a
chaotic collision is still chaotic (LM freezes). But the VELOCITY CHANGE at the
impact is a smooth, algebraic function of the mass ratio:

    m1 * dv1 = -m2 * dv2   (equal-and-opposite collision impulse)
    => m1/m2 = -(dv1 . dv2) / (dv1 . dv1)     (least-squares scalar)
    => density ratio = (m1/m2) * (vol2/vol1)  (volumes known from geometry)

So instead of optimizing, we MEASURE: resolve the impact with a high frame rate
(obs_stride=1), read each object's COM velocity just before and just after
contact, and solve for the mass ratio. Over the ~ms impact window the impulsive
collision force dominates gravity/friction, so the estimate is clean.

Usage: python scripts/recover_density_momentum.py --config configs/collide_hires.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.scene_sim import PER, SceneSim, theta_vec_from_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "collide_hires.yaml"))
    ap.add_argument("--half-window", type=int, default=8,
                    help="samples on each side of impact used to estimate pre/post velocity")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg["sim"].get("obs_stride", 99) <= 2, "needs high-res observation (obs_stride<=2)"
    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])

    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)
        vols = [m / 1000.0 for m in sim.base_mass]  # base_mass = 1000 * volume
        sim.set_theta(theta_true)
        poses = sim.rollout_poses()  # (T, 2, 7), dense
        dt_obs = sim.sim_dt * sim.obs_stride

    com = poses[:, :, :3]                        # (T, 2, 3) COM positions
    vel = np.gradient(com, dt_obs, axis=0)       # (T, 2, 3) COM velocities
    acc = np.gradient(vel, dt_obs, axis=0)       # (T, 2, 3) COM accelerations
    rel_dist = np.linalg.norm(com[:, 0] - com[:, 1], axis=1)
    impact = int(np.argmin(rel_dist))

    # Newton's 3rd law during contact: the horizontal collision forces are
    # equal-and-opposite (gravity is vertical, so it drops out of x,y), so at
    # EVERY instant of contact m0*a0_h = -m1*a1_h. Fit the mass ratio to the
    # horizontal accelerations over the contact spike (least squares) — robust to
    # window edges and the finite soft-contact duration.
    a0h = acc[:, 0, :2]
    a1h = acc[:, 1, :2]
    force_mag = np.linalg.norm(a0h, axis=1) + np.linalg.norm(a1h, axis=1)
    thresh = 0.2 * force_mag.max()               # contact = the acceleration spike
    contact = force_mag > thresh
    print(f"impact at sample {impact} (t={impact * dt_obs * 1000:.1f} ms), "
          f"min center-dist {rel_dist.min():.3f} m, contact samples {contact.sum()}")

    A0 = a0h[contact]; A1 = a1h[contact]
    r_mass = float(-(A0.ravel() @ A1.ravel()) / (A0.ravel() @ A0.ravel() + 1e-12))  # m0/m1
    dens_ratio = r_mass * (vols[1] / vols[0])
    dv1 = vel[min(impact + args.half_window, len(vel) - 1), 0] - vel[max(impact - args.half_window, 0), 0]
    dv2 = vel[min(impact + args.half_window, len(vel) - 1), 1] - vel[max(impact - args.half_window, 0), 1]

    d_true = np.exp(theta_true[0]) / np.exp(theta_true[PER])          # true density ratio
    m_true = (np.exp(theta_true[0]) * vols[0]) / (np.exp(theta_true[PER]) * vols[1])
    print(f"\nΔv obj0 = {np.round(dv1, 3)}  |Δv1|={np.linalg.norm(dv1):.3f}")
    print(f"Δv obj1 = {np.round(dv2, 3)}  |Δv2|={np.linalg.norm(dv2):.3f}")
    print(f"\nmass ratio m0/m1:    measured {r_mass:.3f}   true {m_true:.3f}")
    print(f"DENSITY ratio d0/d1: measured {dens_ratio:.3f}   true {d_true:.3f}   "
          f"({100 * abs(dens_ratio - d_true) / d_true:.1f}% err)")
    ok = abs(dens_ratio - d_true) / d_true < 0.25
    print(f"\n{'RECOVERED via momentum (high-res observation works!)' if ok else 'still off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
