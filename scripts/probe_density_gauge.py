"""Measure how well a scene identifies DENSITY, split into the two gauge directions:

  scale  = all object log-densities up together (a true gauge for gravity+elastic
           contact — invariant, should stay unobservable)
  ratio  = trade density between objects (observable ONLY if contact couples their
           masses — the object-object collision signal)

Reports the ratio/scale curvature ratio: >> 1 means the collision breaks the
per-object density gauge (relative density observable). Also a finite density-swap
test. Use to compare launch-and-settle (M6) vs collision-dominated (M7) scenes.

Usage: python scripts/probe_density_gauge.py --config configs/collide.yaml
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

FD_LOGD = 0.10  # FD step in log-density


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "collide.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n_obj = len(cfg["objects"])
    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    di = [k * PER for k in range(n_obj)]  # log_density indices

    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)

        def traj(th):
            sim.set_theta(th)
            return sim.rollout().ravel()

        r0 = traj(theta_true)

        # Jacobian columns for the density params only
        J = {}
        for idx in di:
            e = np.zeros_like(theta_true); e[idx] = FD_LOGD
            J[idx] = (traj(theta_true + e) - traj(theta_true - e)) / (2 * FD_LOGD)
        n_terms = r0.size / 3

        scale = np.zeros(len(theta_true)); ratio = np.zeros(len(theta_true))
        for k, idx in enumerate(di):
            scale[idx] = 1.0
            ratio[idx] = 1.0 if k == 0 else -1.0
        scale /= np.linalg.norm(scale); ratio /= np.linalg.norm(ratio)

        def curv(direction):
            v = np.zeros(r0.size)
            for idx in di:
                v += direction[idx] * J[idx]
            return float(v @ v) / n_terms

        c_scale, c_ratio = curv(scale), curv(ratio)

        # finite density-swap (only meaningful for 2 objects)
        swap = theta_true.copy()
        if n_obj == 2:
            swap[di[0]], swap[di[1]] = theta_true[di[1]], theta_true[di[0]]
            base = traj(theta_true).reshape(-1, 3)
            swapped = traj(swap).reshape(-1, 3)
            swap_mm = np.linalg.norm(base - swapped, axis=1).max() * 1000

        # per-object density sensitivity (m per unit log-density) vs a velocity
        v_sens = None
        vx_idx = 3  # obj0 v0x
        e = np.zeros_like(theta_true); e[vx_idx] = 0.1
        jvx = (traj(theta_true + e) - traj(theta_true - e)) / 0.2
        v_sens = np.sqrt(float(jvx @ jvx) / n_terms)

    print(f"scene: {Path(args.config).name}  n_obj={n_obj}")
    for k, idx in enumerate(di):
        s = np.sqrt(float(J[idx] @ J[idx]) / n_terms)
        print(f"  obj{k} log-density sensitivity: {s:.4e}  (m per unit log-density)")
    print(f"  reference: obj0 v0x sensitivity:  {v_sens:.4e}")
    print(f"\ndensity SCALE curvature (both up):     {c_scale:.4e}")
    print(f"density RATIO curvature (trade masses): {c_ratio:.4e}")
    print(f"ratio/scale = {c_ratio / max(c_scale, 1e-30):.1f}x")
    if n_obj == 2:
        print(f"density-swap max marker shift: {swap_mm:.1f} mm")
    verdict = ("GAUGE BROKEN: relative density strongly observable" if c_ratio > 8 * c_scale
               else "gauge softened only" if c_ratio > 2 * c_scale else "gauge intact")
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
