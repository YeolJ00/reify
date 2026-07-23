"""Recover the collision scene with CEM (global, gradient-free) instead of LM.

The collision-dominated scene breaks the density gauge (relative density becomes
observable) but makes the loss landscape chaotic -> LM makes zero progress
(scripts/recover_scene.py on configs/collide.yaml). CEM samples the landscape
and follows the underlying trend despite local noise, so it can recover what LM
cannot. This closes the loop on "does the collision let us recover density?"

Usage: python scripts/recover_scene_cem.py --config configs/collide.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.optimize.cem import cem_nd  # noqa: E402
from src.sim.scene_sim import PER, SceneSim, theta_natural, theta_vec_from_cfg  # noqa: E402

SIGMA_OBJ = np.array([0.6, 0.25, 0.2, 0.5, 0.5, 0.5])  # per-object CEM search width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "collide.yaml"))
    ap.add_argument("--pop", type=int, default=28)
    ap.add_argument("--iters", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg["seed"])
    n_obj = len(cfg["objects"])
    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])
    sigma0 = np.tile(SIGMA_OBJ, n_obj)
    names = [f"o{k}.{n}" for k in range(n_obj)
             for n in ["log_density", "mu", "restitution", "v0x", "v0y", "v0z"]]

    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)
        sim.set_theta(theta_true)
        target = sim.rollout()
        F, M = target.shape[0], target.shape[1]
        norm = 0.2 * np.sqrt(F * M)

        def loss_fn(theta):
            sim.set_theta(theta)
            traj = sim.rollout()
            if not np.isfinite(traj).all():
                return 1e6
            r = (traj - target).ravel() / norm
            return float(r @ r)

        print(f"CEM: pop={args.pop} iters={args.iters}  L(init)={loss_fn(theta0):.4e}")
        t0 = time.time()
        theta_hat, hist = cem_nd(loss_fn, mu0=theta0, sigma0=sigma0,
                                 pop_size=args.pop, iters=args.iters, seed=cfg["seed"])
        rms_mm = np.sqrt(loss_fn(theta_hat)) * 0.2 * 1000
        elapsed = time.time() - t0

    nt, nh, n0 = (theta_natural(theta_true, n_obj), theta_natural(theta_hat, n_obj),
                  theta_natural(theta0, n_obj))
    flat = lambda L: [x for o in L for x in [o["density"], o["mu"], o["restitution"], *o["v0"]]]
    ft, fh, f0 = flat(nt), flat(nh), flat(n0)
    print(f"\nCEM best fit RMS {rms_mm:.2f} mm  ({elapsed:.0f}s)")
    print(f"{'param':>14} {'true':>9} {'init':>9} {'recovered':>10}")
    for i, nm in enumerate(names):
        print(f"{nm:>14} {ft[i]:>9.3f} {f0[i]:>9.3f} {fh[i]:>10.3f}")

    # the headline number: recovered density RATIO vs true
    d_true = nt[0]["density"] / nt[1]["density"]
    d_hat = nh[0]["density"] / nh[1]["density"]
    print(f"\ndensity RATIO (obj0/obj1): true {d_true:.2f}  recovered {d_hat:.2f}  "
          f"({'RATIO RECOVERED' if 0.5 * d_true < d_hat < 2 * d_true else 'ratio missed'})")

    out = REPO / "outputs"
    np.savez(out / "scene_cem.npz", theta_true=theta_true, theta_hat=theta_hat,
             theta0=theta0, loss_hist=[h["best_loss"] for h in hist],
             d_true=d_true, d_hat=d_hat)
    print(f"saved {out / 'scene_cem.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
