"""M5.0: forward rigid drop of a real scanned asset. Sanity + density-invariance.

Verifies (a) the object falls, contacts, and settles without tunneling, and
(b) the trajectory is INVARIANT to density (the rigid analog of the cloth
mass-gauge: under gravity, contact dynamics are mass-independent) — an
identifiability prediction we confirm empirically here.

Usage: python scripts/run_rigid_forward.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.rigid_sim import RigidDropSim, theta_vec_from_cfg  # noqa: E402


def main():
    cfg = yaml.safe_load((REPO / "configs" / "drop.yaml").read_text())
    np.random.seed(cfg["seed"])
    wp.init()
    assert wp.get_device(cfg["device"]).is_cuda

    with wp.ScopedDevice(cfg["device"]):
        sim = RigidDropSim(cfg)
        print(f"asset={cfg['asset']['name']}  markers={len(sim.markers_local)}  "
              f"base_mass={sim.base_mass:.4f} kg  frames={sim.num_frames}  dt={sim.sim_dt:.2e}")

        theta = theta_vec_from_cfg(cfg["theta"]["true_values"])
        sim.set_theta(theta)
        t0 = time.time()
        traj = sim.rollout()
        wp.synchronize()
        print(f"rollout {time.time() - t0:.1f} s")

        assert np.isfinite(traj).all(), "non-finite trajectory"
        com = traj.mean(axis=1)  # (F+1,3) approx COM path
        zmin = traj[..., 2].min()
        print(f"COM start z={com[0,2]:.3f}  end z={com[-1,2]:.3f}  min marker z={zmin:.4f}")
        assert zmin > -0.05, "object tunneled through the ground"
        settled = np.linalg.norm(com[-1] - com[-5])
        print(f"COM travel (x,y)={com[-1,0]-com[0,0]:+.3f},{com[-1,1]-com[0,1]:+.3f} m  "
              f"late-frame motion={settled:.4f} m")

        # density-invariance test: same theta but density x4 -> identical markers?
        theta_heavy = theta.copy(); theta_heavy[0] = np.log(np.exp(theta[0]) * 4.0)
        sim.set_theta(theta_heavy)
        traj_heavy = sim.rollout()
        max_diff = np.abs(traj_heavy - traj).max()
        print(f"density x4 max marker-position diff: {max_diff:.6e} m  "
              f"({'INVARIANT (mass-gauge confirmed)' if max_diff < 1e-3 else 'density observable'})")

    out = REPO / "outputs"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "rigid_forward.npz", traj=traj, theta_true=theta,
                        fps=cfg["sim"]["fps"], markers_local=sim.markers_local)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ax = np.arange(len(com)) / cfg["sim"]["fps"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(t_ax, com[:, 0], label="x"); axes[0].plot(t_ax, com[:, 1], label="y")
    axes[0].plot(t_ax, com[:, 2], label="z")
    axes[0].set(xlabel="t (s)", ylabel="COM (m)", title="pose trajectory"); axes[0].legend()
    axes[1].plot(com[:, 0], com[:, 2]); axes[1].scatter(com[0, 0], com[0, 2], c="g", label="start")
    axes[1].scatter(com[-1, 0], com[-1, 2], c="r", label="end")
    axes[1].set(xlabel="x (m)", ylabel="z (m)", title="side view (x-z)"); axes[1].legend()
    axes[1].axhline(0, color="k", lw=0.5)
    for f in range(0, len(traj), max(1, len(traj) // 10)):
        axes[2].scatter(traj[f, :, 0], traj[f, :, 2], s=6, alpha=0.5)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(xlabel="x (m)", ylabel="z (m)", title="markers over time")
    fig.suptitle(f"M5.0 rigid drop — {cfg['asset']['name']}")
    fig.tight_layout(); fig.savefig(out / "rigid_forward.png", dpi=120)
    print(f"saved {out / 'rigid_forward.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
