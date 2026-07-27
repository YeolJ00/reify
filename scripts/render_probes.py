"""Render the three probes of the physics lab as one side-by-side animation:
drop / push-slide / collide, on the same scene. (warp env)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_collide_6dof import _quat_mat  # noqa: E402
from scripts.probe_matrix import GZ, NAMES, PROBES, TRUE, RHO_A, K_CONTACT, DT  # noqa: E402
from src.sim.probe_scene import ProbeScene  # noqa: E402

CAM = {"eye": [0.78, -1.05, 1.14], "target": [0.20, 0.06, 0.76], "fov_deg": 58, "width": 460, "height": 380}
RENDER_STEPS, RSTRIDE = 11000, 150   # ~1.1 s of motion (the matrix only needs 0.28 s)
COLORS = ["#57d7e0", "#f2a53d"]
TITLES = {"drop": "1 · Drop  →  bounciness", "slide": "2 · Push  →  friction",
          "collide": "3 · Collide  →  mass ratio"}


def main():
    wp.init()
    frames = {}
    with wp.ScopedDevice("cuda:0"):
        mk = lambda p0_, v0_: ProbeScene(NAMES, p0_, v0_, densities=(RHO_A, RHO_A * TRUE["ratio"]),
                                        cd=TRUE["cd"], mu=TRUE["mu"], ground_z=GZ,
                                        k=K_CONTACT, dt=DT, n_steps=RENDER_STEPS)
        p0 = mk([[0, 0, GZ + 0.2]] * 2, [[0, 0, 0]] * 2)
        cl, bd, r = p0.center_local.numpy(), p0.body.numpy(), p0.radius
        rest = [float(GZ + r - cl[bd == i][:, 2].min()) for i in range(2)]
        layouts = {
            "drop":    ([[0.0, 0.0, GZ + 0.14], [0.40, 0.40, rest[1]]], [[0, 0, 0], [0, 0, 0]]),
            "slide":   ([[-0.12, 0.0, rest[0]], [0.40, 0.40, rest[1]]], [[0.9, 0, 0], [0, 0, 0]]),
            "collide": ([[-0.10, 0.0, rest[0]], [0.08, 0.0, rest[1]]], [[1.6, 0, 0], [0, 0, 0]]),
        }
        for pr in PROBES:
            sc = mk(*layouts[pr])
            sc.rollout()
            P = np.stack([sc.pos[t].numpy() for t in range(0, sc.n_steps + 1, RSTRIDE)])
            Q = np.stack([sc.rot[t].numpy() for t in range(0, sc.n_steps + 1, RSTRIDE)])
            frames[pr] = (P, Q)
        print("rolled out", {k: v[0].shape for k, v in frames.items()})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageSequence
    cam = Camera(CAM)
    T = min(len(frames[p][0]) for p in PROBES)
    step = max(1, T // 45)
    idx = list(range(0, T, step))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), facecolor="#141210")

    def draw(k):
        t = idx[k]
        for ax, pr in zip(axes, PROBES):
            ax.clear(); ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0)
            ax.axis("off"); ax.set_facecolor("#141210")
            g, _ = cam.project(np.array([[-.35, -.35, GZ], [.95, -.35, GZ], [.95, .60, GZ], [-.35, .60, GZ]]))
            ax.fill(g[:, 0], g[:, 1], color="#3a3128", zorder=-1)
            P, Q = frames[pr]
            pts, cols = [], []
            for b in range(P.shape[1]):
                w = P[t, b] + cl[bd == b] @ _quat_mat(Q[t, b]).T
                pts.append(w); cols += [COLORS[b]] * len(w)
            w = np.concatenate(pts); uv, dep = cam.project(w)
            order = np.argsort(-dep)
            sz = (cam.fx * r / np.clip(dep, 1e-3, None)) ** 2 * 3.0
            ax.scatter(uv[order, 0], uv[order, 1], s=sz[order],
                       c=[cols[i] for i in order], alpha=0.95, edgecolors="none")
            ax.set_title(TITLES[pr], color="#f2ece2", fontsize=11, weight="bold")
        return []

    FuncAnimation(fig, draw, frames=len(idx), interval=90).save(
        REPO / "outputs" / "matrix" / "probes.gif", writer=PillowWriter(fps=11))
    p = REPO / "outputs" / "matrix" / "probes.gif"
    im = Image.open(p); fs = [f.convert("P", palette=Image.ADAPTIVE, colors=80) for f in ImageSequence.Iterator(im)]
    fs[0].save(p, save_all=True, append_images=fs[1:], loop=0, duration=90, optimize=True)
    print(f"wrote {p} ({p.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
