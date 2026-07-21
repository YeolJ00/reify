"""M0: one forward cloth-flag rollout on GPU with constant wind.

Saves the vertex trajectory to outputs/trajectory_<solver>.npz and a summary
plot to outputs/forward_<solver>.png, and prints sanity numbers.

Usage: python scripts/run_forward.py [--config configs/flag.yaml] [--solver vbd|semi_implicit]
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

from src.sim.rollout import FlagSim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "flag.yaml"))
    ap.add_argument("--solver", default=None, help="override sim.solver from config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.solver:
        cfg["sim"]["solver"] = args.solver
    np.random.seed(cfg["seed"])

    wp.init()
    device = wp.get_device(cfg["device"])
    print(f"device: {device}  (is_cuda={device.is_cuda})")
    assert device.is_cuda, "M0 requires a GPU"

    with wp.ScopedDevice(cfg["device"]):
        sim = FlagSim(cfg, requires_grad=False)
        print(
            f"solver={sim.solver_name}  particles={sim.model.particle_count}  "
            f"tris={sim.model.tri_count}  frames={sim.num_frames}  "
            f"substeps={sim.substeps}  dt={sim.sim_dt:.5f}"
        )

        t0 = time.time()
        sim.rollout()
        wp.synchronize()
        elapsed = time.time() - t0

        traj = sim.trajectory()  # (F+1, N, 3)

    assert np.isfinite(traj).all(), "trajectory contains NaN/inf — sim exploded"

    # sanity numbers: the free (fly) end should move; pinned edge should not
    disp = np.linalg.norm(traj - traj[0], axis=2)  # (F+1, N)
    dim_x, dim_y = cfg["cloth"]["dim_x"], cfg["cloth"]["dim_y"]
    # grid indexing from add_cloth_grid (builder source): index = y * (dim_x+1) + x
    pinned = np.arange(0, dim_y + 1) * (dim_x + 1)  # x = 0 column (fix_left)
    fly_tip = (dim_y // 2) * (dim_x + 1) + dim_x  # far-edge midpoint

    print(f"rollout wall time: {elapsed:.2f} s  ({sim.num_frames * sim.substeps} substeps)")
    print(f"max displacement (any vertex): {disp.max():.4f} m")
    print(f"pinned-edge max displacement:  {disp[:, pinned].max():.6f} m (should be ~0)")
    print(f"fly-tip final displacement:    {disp[-1, fly_tip]:.4f} m")

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    npz_path = out / f"trajectory_{sim.solver_name}.npz"
    np.savez_compressed(
        npz_path,
        trajectory=traj,
        wind_strength=cfg["wind"]["strength"],
        wind_direction=cfg["wind"]["direction"],
        fps=cfg["sim"]["fps"],
    )
    print(f"saved trajectory: {npz_path}  shape={traj.shape}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_axis = np.arange(traj.shape[0]) / cfg["sim"]["fps"]
    axes[0].plot(t_axis, disp.mean(axis=1), label="mean")
    axes[0].plot(t_axis, disp[:, fly_tip], label="fly tip")
    axes[0].set(xlabel="time (s)", ylabel="displacement (m)", title="vertex displacement")
    axes[0].legend()
    for k, (a, b, lbl) in enumerate([(0, 1, "XY (top)"), (0, 2, "XZ (side)")]):
        ax = axes[k + 1]
        for f in range(0, traj.shape[0], max(1, traj.shape[0] // 8)):
            ax.scatter(traj[f, :, a], traj[f, :, b], s=1, alpha=0.35)
        ax.scatter(traj[-1, :, a], traj[-1, :, b], s=2, color="k")
        ax.set(xlabel="xyz"[a], ylabel="xyz"[b], title=f"cloth over time, {lbl}")
        ax.set_aspect("equal")
    fig.suptitle(f"M0 forward rollout — solver={sim.solver_name}, wind={cfg['wind']['strength']}")
    fig.tight_layout()
    png_path = out / f"forward_{sim.solver_name}.png"
    fig.savefig(png_path, dpi=120)
    print(f"saved plot: {png_path}")


if __name__ == "__main__":
    main()
