"""Render the real multi-asset scene: the wooden-table mesh with both scanned
objects colliding on top, at several frames. Proves the real scene runs in
Newton's contact solver. Composites all meshes with global painter-sort.

Usage: python scripts/render_scene.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.data.assets import decimate, load_asset  # noqa: E402
from src.render.camera import Camera  # noqa: E402
from src.sim.scene_sim import SceneSim, _rotate, theta_vec_from_cfg  # noqa: E402


def main():
    cfg = yaml.safe_load((REPO / "configs" / "scene.yaml").read_text())
    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)
        sim.set_theta(theta_vec_from_cfg(cfg["theta"]["true_values"]))
        poses = sim.rollout_poses()  # (F+1, N, 7)

    # real table mesh, lifted so its top sits at table.top_z
    t = cfg["table"]
    tt = decimate(load_asset("scenes", t["name"]), int(t["target_faces"]))
    tv = tt.vertices.copy()
    tv[:, 2] += float(t["top_z"]) - tv[:, 2].max()  # align mesh top to tabletop height
    table = (tv, tt.faces, np.array([0.62, 0.46, 0.30]))  # wood color
    obj_meshes = [(sim.mesh_verts[i], sim.mesh_faces[i], np.array([0.80, 0.80, 0.84]))
                  for i in range(sim.n_obj)]

    cam = Camera({"eye": [1.25, -1.45, 1.15], "target": [-0.02, 0.0, 0.72],
                  "fov_deg": 40, "width": 460, "height": 420})
    light = np.array([0.4, -0.5, 0.85]); light /= np.linalg.norm(light)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    n_show = 6
    idx = np.linspace(0, len(poses) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.3 * n_show, 2.6))

    for ax, f in zip(axes, idx):
        tris, depths, cols = [], [], []
        # static table
        for V, Fc, base in [table]:
            uv, dep = cam.project(V)
            _accum(tris, depths, cols, V, Fc, uv, dep, base, light)
        # objects at this frame
        for i, (V0, Fc, base) in enumerate(obj_meshes):
            p, q = poses[f, i, :3], poses[f, i, 3:]
            V = p + _rotate(q, V0)
            uv, dep = cam.project(V)
            _accum(tris, depths, cols, V, Fc, uv, dep, base, light)
        order = np.argsort(-np.array(depths))
        pc = PolyCollection([tris[k] for k in order], facecolors=[cols[k] for k in order],
                            edgecolors="none")
        ax.add_collection(pc)
        ax.set_xlim(0, cam.width); ax.set_ylim(cam.height, 0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"t={f / cfg['sim']['fps']:.2f}s", fontsize=9)

    fig.suptitle("M6 — two real scanned objects colliding on a real table (Newton contact sim)",
                 fontsize=11)
    fig.tight_layout()
    o = REPO / "outputs" / "scene_render.png"
    fig.savefig(o, dpi=120, facecolor="white")
    print(f"saved {o}")


def _accum(tris, depths, cols, V, F, uv, dep, base, light):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    shade = np.clip(0.3 + 0.7 * np.abs(n @ light), 0, 1)
    tz = dep[F].mean(1)
    for k in range(len(F)):
        tris.append(uv[F[k]]); depths.append(tz[k]); cols.append(base * shade[k])


if __name__ == "__main__":
    wp.init()
    main()
