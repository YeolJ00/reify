"""Render the squishy cloth splat as an animated GIF + a floppy-vs-stiff comparison.
A tilted cloth sheet drops onto the table and collapses; stiffness = squishiness.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_splat import DiffSplat  # noqa: E402

KW = dict(dim=16, cell=0.035, mass=0.04, height=0.95, ground_z=0.706, tilt_deg=55,
          tri_kd=8.0, edge_ke=2.0, k=2500.0, cd=25.0, mu=0.5, num_frames=60, substeps=48)
CAM = {"eye": [0.62, -0.62, 1.02], "target": [0.0, 0.0, 0.76], "fov_deg": 46, "width": 620, "height": 480}
CLOTH = np.array([0.78, 0.24, 0.52])   # magenta fabric


def run(ke):
    with wp.ScopedDevice("cuda:0"):
        s = DiffSplat(tri_ke=ke, requires_grad=False, **KW)
        s.rollout()
        return s.trajectory(), s.model.tri_indices.numpy()


def draw_cloth(ax, verts, F, cam, light):
    from matplotlib.collections import PolyCollection
    uv, dep = cam.project(verts)
    n = np.cross(verts[F[:, 1]] - verts[F[:, 0]], verts[F[:, 2]] - verts[F[:, 0]])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    sh = np.clip(0.35 + 0.65 * np.abs(n @ light), 0, 1)
    tz = dep[F].mean(1); order = np.argsort(-tz)
    ax.add_collection(PolyCollection([uv[F[k]] for k in order],
                      facecolors=[CLOTH * sh[k] for k in order],
                      edgecolors=(0, 0, 0, 0.25), linewidths=0.4))
    g, _ = cam.project(np.array([[-0.35, -0.35, 0.706], [0.35, -0.35, 0.706],
                                 [0.35, 0.35, 0.706], [-0.35, 0.35, 0.706]]))
    ax.fill(g[:, 0], g[:, 1], color="#3a3128", zorder=-1)


def main():
    wp.init()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    cam = Camera(CAM); light = np.array([0.4, -0.5, 0.85]); light /= np.linalg.norm(light)
    W, H = cam.width, cam.height

    traj, F = run(300.0)          # floppy -> squishy splat
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    idx = np.linspace(0, len(traj) - 1, 48).astype(int)

    def frame(i):
        ax.clear(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.set_facecolor("#141210")
        draw_cloth(ax, traj[idx[i]], F, cam, light)
        ax.text(0.03, 0.05, "squishy cloth splat — floppy fabric", transform=ax.transAxes,
                color="#c9bfae", fontsize=10)
        return []
    FuncAnimation(fig, frame, frames=len(idx), interval=60).save(
        REPO / "outputs" / "splat.gif", writer=PillowWriter(fps=16))
    plt.close(fig)
    print("saved outputs/splat.gif")

    # floppy vs stiff during active collapse (when they differ most)
    stiff, _ = run(6000.0)
    k = int(len(traj) * 0.42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#141210")
    for ax, tr, name, c in [(axes[0], traj, "Floppy (squishy)", "#57d7e0"),
                            (axes[1], stiff, "Stiff", "#f2a53d")]:
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.set_facecolor("#141210")
        draw_cloth(ax, tr[k], F, cam, light)
        ax.set_title(name, color=c, fontsize=14, weight="bold")
    fig.suptitle("Same drop, mid-collapse — floppy crumples, stiff resists",
                 color="#f2ece2", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(REPO / "outputs" / "splat_compare.png", dpi=120, facecolor="#141210")
    print("saved outputs/splat_compare.png")


if __name__ == "__main__":
    main()
