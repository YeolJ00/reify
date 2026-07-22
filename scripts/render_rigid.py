"""Render the real scanned asset falling & bouncing — a contact sheet proving a
real GSO mesh runs in a real Newton contact sim. Projects the decimated teapot
mesh (shaded by camera-facing normal, painter-sorted) through our pinhole camera
at several frames of the drop.

Usage: python scripts/render_rigid.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.rigid_sim import RigidDropSim, _rotate, theta_vec_from_cfg  # noqa: E402


def main():
    cfg = yaml.safe_load((REPO / "configs" / "drop.yaml").read_text())
    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        sim = RigidDropSim(cfg)
        sim.set_theta(theta_vec_from_cfg(cfg["theta"]["true_values"]))
        poses = sim.rollout_poses()
    V, F = sim.mesh_verts, sim.mesh_faces

    # camera framing the drop volume, looking slightly down the +x travel
    cam = Camera({"eye": [1.1, -1.3, 0.7], "target": [0.15, 0.05, 0.12],
                  "fov_deg": 42, "width": 420, "height": 420})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    n_show = 8
    idx = np.linspace(0, len(poses) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.05 * n_show, 2.5))
    light = np.array([0.4, -0.5, 0.8]); light = light / np.linalg.norm(light)

    for ax, f in zip(axes, idx):
        p, q = poses[f][:3], poses[f][3:]
        world = p + _rotate(q, V)                       # (Nv,3)
        uv, depth = cam.project(world)
        tri = uv[F]                                      # (Nf,3,2)
        tri_z = depth[F].mean(1)
        # face normal for shading
        n = np.cross(world[F[:, 1]] - world[F[:, 0]], world[F[:, 2]] - world[F[:, 0]])
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        shade = np.clip(0.25 + 0.75 * np.abs(n @ light), 0, 1)
        order = np.argsort(-tri_z)                       # painter: far first
        colors = plt.cm.bone(shade[order])
        pc = PolyCollection(tri[order], facecolors=colors, edgecolors="none")
        ax.add_collection(pc)
        # ground line at z=0
        g0, _ = cam.project(np.array([[-.3, .0, 0], [.6, .0, 0]]))
        ax.plot(g0[:, 0], g0[:, 1], color="0.7", lw=1)
        ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"t={f / cfg['sim']['fps']:.2f}s", fontsize=9)

    fig.suptitle(f"M5 — real scanned {cfg['asset']['name']} dropped in Newton (fall → bounce → settle)",
                 fontsize=11)
    fig.tight_layout()
    out = REPO / "outputs" / "rigid_drop_render.png"
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
