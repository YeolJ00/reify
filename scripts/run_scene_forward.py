"""M6.0: forward multi-asset scene. Two real scanned objects launched toward each
other on a real table, colliding and settling. Sanity + a collision-coupling test:
does object-object contact make RELATIVE density observable (unlike the single-
object drop where density is a pure gauge)?

Usage: python scripts/run_scene_forward.py [--env table|ground]
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

from src.sim.scene_sim import SceneSim, theta_vec_from_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="override environment (table|ground)")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "scene.yaml").read_text())
    if args.env:
        cfg["environment"] = args.env
    np.random.seed(cfg["seed"])
    wp.init()
    assert wp.get_device(cfg["device"]).is_cuda

    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)
        print(f"env={cfg['environment']}  n_obj={sim.n_obj}  "
              f"base_masses={[round(m, 3) for m in sim.base_mass]} kg  frames={sim.num_frames}")
        theta = theta_vec_from_cfg(cfg["theta"]["true_values"])
        sim.set_theta(theta)
        t0 = time.time()
        poses = sim.rollout_poses()
        wp.synchronize()
        print(f"rollout {time.time() - t0:.1f} s  poses {poses.shape}")
        assert np.isfinite(poses).all(), "scene exploded (non-finite)"

        for i in range(sim.n_obj):
            p = poses[:, i, :3]
            print(f"  obj{i} {cfg['objects'][i]['name'][:28]:28s} "
                  f"start={np.round(p[0], 3)} end={np.round(p[-1], 3)} min_z={p[:, 2].min():.3f}")
        d = np.linalg.norm(poses[:, 0, :3] - poses[:, 1, :3], axis=1)
        collided = d.min() < 0.6 * d[0]
        print(f"inter-object dist: start {d[0]:.3f}  min {d.min():.3f}  end {d[-1]:.3f}  "
              f"-> {'COLLISION' if collided else 'no collision'}")

        # collision-coupling test: swap the two densities (keep total), re-roll.
        # If markers change, the RELATIVE density is observable (gauge broken).
        th_swap = theta.copy()
        th_swap[0], th_swap[6] = theta[6], theta[0]  # swap log_density of obj0/obj1
        sim.set_theta(th_swap)
        poses_swap = sim.rollout_poses()
        m = sim.rollout  # noqa
        traj = sim.rollout()  # markers at true theta already set? re-set
        sim.set_theta(theta); base = sim.rollout()
        sim.set_theta(th_swap); swapped = sim.rollout()
        diff = np.abs(base - swapped).max()
        print(f"density-swap max marker diff: {diff:.4f} m  "
              f"({'RELATIVE DENSITY OBSERVABLE (collision breaks gauge)' if diff > 5e-3 else 'still gauge'})")

    out = REPO / "outputs"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "scene_forward.npz", poses=poses, theta_true=theta,
                        n_obj=sim.n_obj, fps=cfg["sim"]["fps"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tax = np.arange(poses.shape[0]) / cfg["sim"]["fps"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for i in range(sim.n_obj):
        axes[0].plot(tax, poses[:, i, 0], label=f"obj{i} x")
        axes[0].plot(tax, poses[:, i, 2], ls="--", label=f"obj{i} z")
    axes[0].set(xlabel="t (s)", ylabel="pos (m)", title="object positions"); axes[0].legend(fontsize=7)
    axes[1].plot(tax, d); axes[1].set(xlabel="t (s)", ylabel="center dist (m)", title="inter-object distance")
    for i in range(sim.n_obj):
        axes[2].plot(poses[:, i, 0], poses[:, i, 2], label=f"obj{i}")
        axes[2].scatter(poses[0, i, 0], poses[0, i, 2], c="g"); axes[2].scatter(poses[-1, i, 0], poses[-1, i, 2], c="r")
    axes[2].set(xlabel="x (m)", ylabel="z (m)", title="paths (x-z)"); axes[2].legend(fontsize=7)
    fig.suptitle(f"M6.0 multi-asset scene ({cfg['environment']})")
    fig.tight_layout(); fig.savefig(out / "scene_forward.png", dpi=120)
    print(f"saved {out / 'scene_forward.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
