"""Render the 6-DOF collision as an animated GIF: the two real scanned meshes
translate and TUMBLE through the collision, shaded and depth-sorted.

Usage: python scripts/render_collision_video.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.data.assets import decimate, load_asset  # noqa: E402
from src.render.camera import Camera  # noqa: E402
from src.sim.diff_collide_6dof import DiffCollide6DOF, _quat_mat  # noqa: E402

NAMES = ["Great_Dinos_Triceratops_Toy", "Great_Dinos_Triceratops_Toy"]
POS0 = [[-0.12, -0.015, 0.0], [0.12, 0.015, 0.0]]
VEL0 = [[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]
ANG0 = [[0.0, 0.0, 14.0], [0.0, 0.0, 0.0]]
COLORS = [np.array([0.30, 0.45, 0.85]), np.array([0.90, 0.55, 0.20])]  # heavy / light


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        sim = DiffCollide6DOF(NAMES, POS0, VEL0, ang0=ANG0, pitch=0.012, dt=2.0e-4,
                              n_steps=1800, k=3000.0, cd=12.0, mu=0.55, requires_grad=False)
        sim.set_log_density(np.log([800.0, 400.0]))
        sim.rollout()
        pos = sim.positions()
        ori = sim.orientations()

    tm = decimate(load_asset("rigid", NAMES[0]), 500)
    V = (tm.vertices - tm.vertices.mean(0)).astype(np.float64)
    F = tm.faces.astype(np.int32)

    cam = Camera({"eye": [0.28, -0.42, 0.28], "target": [0.0, 0.0, 0.0],
                  "fov_deg": 40, "width": 640, "height": 460})
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection

    n_frames = 60
    idx = np.linspace(0, len(pos) - 1, n_frames).astype(int)
    fig = plt.figure(figsize=(6.4, 4.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    def draw(fi):
        ax.clear()
        ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0)
        ax.axis("off"); ax.set_facecolor("white")
        f = idx[fi]
        tris, depths, cols = [], [], []
        for b in range(2):
            R = _quat_mat(ori[f, b])
            world = pos[f, b] + V @ R.T
            uv, dep = cam.project(world)
            n = np.cross(world[F[:, 1]] - world[F[:, 0]], world[F[:, 2]] - world[F[:, 0]])
            n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
            shade = np.clip(0.35 + 0.65 * np.abs(n @ light), 0, 1)
            tz = dep[F].mean(1)
            for kk in range(len(F)):
                tris.append(uv[F[kk]]); depths.append(tz[kk]); cols.append(COLORS[b] * shade[kk])
        order = np.argsort(-np.array(depths))
        ax.add_collection(PolyCollection([tris[k] for k in order],
                                         facecolors=[cols[k] for k in order], edgecolors="none"))
        # ground line
        g, _ = cam.project(np.array([[-0.25, 0, 0], [0.25, 0, 0]]))
        ax.plot(g[:, 0], g[:, 1], color="0.8", lw=1)
        ax.text(0.02, 0.06, "two real scanned objects — 6-DOF collision with friction",
                transform=ax.transAxes, fontsize=9, color="0.3")
        return ax.collections

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=50)
    out = REPO / "outputs" / "collision.gif"
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved {out}  ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
