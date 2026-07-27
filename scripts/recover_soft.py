"""Recover a volumetric soft body's stiffness from how it squashes, and render the
jelly deforming. A soft cube (low k_mu) squashes ~30% and jiggles; a stiff one barely
dents. We recover k_mu from a target rollout by CEM (gradients through the extended
soft-contact are unreliable, as with the cloth splat). (warp env.)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_soft import DiffSoft  # noqa: E402

GROUND_Z = 0.706
CAM = {"eye": [0.34, -0.34, 0.90], "target": [0.0, 0.0, 0.74], "fov_deg": 44, "width": 560, "height": 480}
JELLY = np.array([0.45, 0.78, 0.62])   # green jelly


def rollout(k_mu):
    with wp.ScopedDevice("cuda:0"):
        s = DiffSoft(k_mu=k_mu, requires_grad=False)
        s.rollout()
        return s.trajectory(), s.tris


def main():
    wp.init()
    K_TRUE = 3000.0                       # soft jelly to recover
    target, tris = rollout(K_TRUE)
    stiff, _ = rollout(30000.0)           # a stiff cube for contrast

    # --- CEM recovery of log(k_mu) from the squash trajectory ---
    idx = np.linspace(0, len(target) - 1, 15).astype(int)
    tgt = target[idx]
    def loss(kmu):
        tr, _ = rollout(kmu)
        return float(np.mean((tr[idx] - tgt) ** 2))
    print(f"recovering k_mu (true {K_TRUE:.0f}) ...")
    # coarse log-grid to seed (avoids CEM collapsing into a shallow local basin)
    grid = np.geomspace(1500.0, 20000.0, 8)
    gl = np.array([loss(c) for c in grid])
    m = np.log(grid[np.argmin(gl)]); sd = 0.35
    print(f"  grid best k_mu={grid[np.argmin(gl)]:.0f}  loss={gl.min():.2e}")
    rng = np.random.default_rng(0)
    for it in range(4):                   # CEM refine around the grid seed
        cand = np.exp(m + sd * rng.standard_normal(8))
        ls = np.array([loss(c) for c in cand])
        elite = cand[np.argsort(ls)[:3]]
        m, sd = np.log(elite).mean(), max(np.log(elite).std(), 0.03)
        print(f"  round {it}: best k_mu={elite[0]:.0f}  loss={ls.min():.2e}")
    k_hat = float(np.exp(m))
    print(f"\nRECOVERED k_mu = {k_hat:.0f}  vs true {K_TRUE:.0f}  ({100*abs(k_hat-K_TRUE)/K_TRUE:.1f}%)")

    _render(target, stiff, tris)
    return 0


def _render(soft, stiff, tris):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageSequence
    cam = Camera(CAM); light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    W, H = cam.width, cam.height
    out = REPO / "outputs" / "soft"; out.mkdir(parents=True, exist_ok=True)

    def draw(ax, verts, color):
        uv, dep = cam.project(verts)
        n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        sh = np.clip(0.4 + 0.6 * np.abs(n @ light), 0, 1); order = np.argsort(-dep[tris].mean(1))
        ax.add_collection(PolyCollection([uv[tris[i]] for i in order], facecolors=[color * sh[i] for i in order],
                          edgecolors=(0, 0, 0, 0.15), linewidths=0.3))
        g, _ = cam.project(np.array([[-.3, -.3, GROUND_Z], [.3, -.3, GROUND_Z], [.3, .3, GROUND_Z], [-.3, .3, GROUND_Z]]))
        ax.fill(g[:, 0], g[:, 1], color="#3a3128", zorder=-1)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.set_facecolor("#141210")

    # soft jelly squashing (the money video)
    idx = np.linspace(0, len(soft) - 1, 34).astype(int)
    fig = plt.figure(figsize=(W / 130, H / 130), dpi=120); ax = fig.add_axes([0, 0, 1, 1])
    def fr(i):
        ax.clear(); draw(ax, soft[idx[i]], JELLY)
        ax.text(0.04, 0.06, "soft body — squashes and jiggles", transform=ax.transAxes, color="#c9bfae", fontsize=10)
        return []
    FuncAnimation(fig, fr, frames=len(idx), interval=70).save(out / "soft_jelly.gif", writer=PillowWriter(fps=16))
    im = Image.open(out / "soft_jelly.gif"); fs = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in ImageSequence.Iterator(im)]
    fs[0].save(out / "soft_jelly.gif", save_all=True, append_images=fs[1:], loop=0, duration=70, optimize=True)

    # soft vs stiff at max squash
    ks = int(np.argmin(soft[:, :, 2].max(1) - soft[:, :, 2].min(1)))
    fig2, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor="#141210")
    for a, tr, name, col in [(axes[0], soft, "Soft (k_mu 3k) — squashes ~30%", JELLY),
                             (axes[1], stiff, "Stiff (k_mu 30k) — barely dents", np.array([0.5, 0.7, 0.85]))]:
        draw(a, tr[ks], col); a.set_title(name, color="#f2ece2", fontsize=12, weight="bold")
    fig2.suptitle("Volumetric soft body — stiffness read from how the cube deforms", color="#f2ece2", fontsize=13, weight="bold")
    fig2.tight_layout(); fig2.savefig(out / "soft_compare.png", dpi=120, facecolor="#141210")
    print(f"wrote {out}/soft_jelly.gif + soft_compare.png")


if __name__ == "__main__":
    raise SystemExit(main())
