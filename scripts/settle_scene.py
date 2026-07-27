"""Settle a roughly-placed multi-object scene into a sim-ready resting state.

Places 3 real scanned assets above a table — floating, and two interpenetrating —
then runs the settling pass (src/sim/settle.py) until they rest stably with no
overlap. Prints penetration + kinetic-energy before/after and writes a before/after
figure + a settling animation. (warp env.)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_collide_6dof import _quat_mat  # noqa: E402
from src.sim.settle import SettleScene  # noqa: E402

GROUND_Z = 0.706
NAMES = ["Threshold_Porcelain_Teapot_White", "Schleich_Lion_Action_Figure", "Great_Dinos_Triceratops_Toy"]
POS0 = [[0.00, 0.00, 0.90],     # teapot — floating
        [0.035, 0.02, 0.87],    # lion   — floating AND cores interpenetrating the teapot
        [-0.11, -0.02, 0.84]]   # triceratops — floating
COLORS = ["#57d7e0", "#f2a53d", "#d64a94"]
CAM = {"eye": [0.85, -0.85, 1.02], "target": [-0.02, 0.0, 0.78], "fov_deg": 50, "width": 720, "height": 540}


def world_spheres(scene, pos, rot):
    cl = scene.center_local.numpy(); body = scene.body.numpy()
    out = np.empty_like(cl)
    for b in range(scene.N):
        m = body == b
        out[m] = pos[b] + cl[m] @ _quat_mat(rot[b]).T
    return out, body


def draw(ax, scene, pos, rot, cam):
    ax.clear(); ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0); ax.axis("off"); ax.set_facecolor("#141210")
    g, _ = cam.project(np.array([[-0.4, -0.4, GROUND_Z], [0.4, -0.4, GROUND_Z], [0.4, 0.4, GROUND_Z], [-0.4, 0.4, GROUND_Z]]))
    ax.fill(g[:, 0], g[:, 1], color="#3a3128", zorder=-1)
    w, body = world_spheres(scene, pos, rot); uv, dep = cam.project(w)
    order = np.argsort(-dep)
    r = scene.radius.numpy()
    sz = (cam.fx * r / np.clip(dep, 1e-3, None)) ** 2 * 3.1
    for k in order:
        ax.scatter(uv[k, 0], uv[k, 1], s=sz[k], c=COLORS[body[k]], alpha=0.9, edgecolors="none")


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        scene = SettleScene(NAMES, POS0, ground_z=GROUND_Z)
        pos_i, rot_i = scene.poses()
        pen_i = scene.metrics()
        print(f"initial : max penetration {pen_i:.1f} mm")
        snaps = scene.run(n_iters=2200, snapshot_every=30)
        pen_f = scene.metrics()
        pos_f, rot_f = scene.poses()
        motion = np.abs(pos_f - snaps[-2][0]).max() * 1000.0
        print(f"settled : max penetration {pen_f:.1f} mm | last-iter motion {motion:.3f} mm")
        for b, name in enumerate(NAMES):
            print(f"  {name:34s} z: {pos_i[b][2]:.3f} -> {pos_f[b][2]:.3f} m  (dropped {pos_i[b][2]-pos_f[b][2]:+.3f})")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    cam = Camera(CAM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#141210")
    draw(axes[0], scene, pos_i, rot_i, cam); axes[0].set_title(f"rough placement — {pen_i:.0f} mm overlap, floating", color="#ef6a5a", fontsize=13, weight="bold")
    draw(axes[1], scene, pos_f, rot_f, cam); axes[1].set_title(f"after settling — {pen_f:.1f} mm overlap, at rest", color="#7fd18b", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(REPO / "outputs" / "settle_before_after.png", dpi=115, facecolor="#141210")
    print("wrote outputs/settle_before_after.png")

    figA = plt.figure(figsize=(cam.width / 130, cam.height / 130), dpi=110); axA = figA.add_axes([0, 0, 1, 1])
    def frame(i):
        p, r = snaps[i]; draw(axA, scene, p, r, cam)
        axA.text(0.03, 0.05, "settling roughly-placed assets", transform=axA.transAxes, color="#c9bfae", fontsize=10)
        return []
    FuncAnimation(figA, frame, frames=len(snaps), interval=80).save(REPO / "outputs" / "settle.gif", writer=PillowWriter(fps=14))
    print("wrote outputs/settle.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
