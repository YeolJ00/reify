"""Recover a ball's 3D physics from a Wan-generated video, the proper way:
CoTracker gives robust 2D tracks, then a differentiable sphere sim is fit so its
CAMERA-PROJECTED centre matches the tracked centroid (architecture steps 6-8).
Physics + projection disentangle the real fall/bounce/roll from monocular image
motion, recovering launch velocity, restitution (via damping cd) and friction.

Run (video env for CoTracker, then warp env for the fit) — this script assumes the
CoTracker centroid is cached; produces recovered params + an overlay. (warp env.)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_sphere import DiffSphere, _proj_loss  # noqa: E402

CAM = {"eye": [0.9, -0.9, 1.12], "target": [0.13, 0.0, 0.95], "fov_deg": 46, "width": 544, "height": 448}
P0 = [0.13, 0.0, 1.16]; R = 0.075; GROUND_Z = 0.706


def main():
    frames = np.load(REPO / "outputs" / "i2v_wan5b_seed0.npz")["frames"]
    d = np.load(REPO / "outputs" / "wan_ball" / "cotrack_seed0.npz"); cen = d["cen"]; vis = d["vis"]
    F = len(frames); cam = Camera(CAM)

    # release = first frame the centroid has clearly started to fall
    y = cen[:, 1]
    trel = next((t for t in range(F - 2) if y[t] - y[:6].mean() > 12 and y[t + 1] > y[t]), 7)
    # fit only the first PHYSICAL arc: release -> first front arrival (ball lowest in
    # image). Beyond it the ball rolls back and forth in depth, which a flat table can't
    # reproduce (Wan's non-physical part) — we quantify that separately.
    t_end = trel + int(np.argmax(y[trel:trel + 17]))
    t_end = max(t_end, trel + 8)
    nf = t_end - trel + 1
    tgt = cen[trel:t_end + 1]
    w = vis[trel:t_end + 1].mean(1)
    print(f"release frame {trel}, first front arrival {t_end}, fitting {nf}-frame physical arc")

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        Rc, tc, fx, fy, cx, cy = cam.wp_args()
        tgt_wp = [wp.array([wp.vec2(float(tgt[i][0]), float(tgt[i][1]))], dtype=wp.vec2) for i in range(nf)]
        norm = CAM["width"] * np.sqrt(nf)
        sim = DiffSphere(P0, R, GROUND_Z, nf, substeps=25, v0=(0, 0, 0), cd=8.0, mu=0.4, requires_grad=True)

        def fwd(v0, cd, mu):
            sim.set_v0(v0); sim.set_cd(cd); sim.set_mu(mu)
            sim.rollout(); sim.loss.zero_()
            sc = 1.0 / (norm * norm)
            for i in range(nf):
                wp.launch(_proj_loss, 1,
                          inputs=[sim.pos[i * sim.substeps], tgt_wp[i], float(w[i]),
                                  Rc, tc, fx, fy, cx, cy, sc], outputs=[sim.loss])

        # ---- multi-start Adam (gradient through contact) ----
        rng = np.random.default_rng(0)
        starts = [np.array([0, 0, 0, np.log(8.0), 0.4])] + \
                 [np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 0),
                            np.log(rng.uniform(3, 25)), rng.uniform(0.05, 0.8)]) for _ in range(5)]
        best = (np.inf, None)
        for si, th0 in enumerate(starts):
            th = th0.copy(); m = np.zeros(5); v = np.zeros(5); bloc = (np.inf, th.copy())
            for it in range(70):
                tape = wp.Tape()
                with tape:
                    fwd(th[:3], np.exp(th[3]), th[4])
                tape.backward(sim.loss)
                L = float(sim.loss.numpy()[0])
                gv = sim.v0.grad.numpy()[0]; gcd = float(sim.cd.grad.numpy()[0]) * np.exp(th[3])
                gmu = float(sim.mu.grad.numpy()[0])
                g = np.array([gv[0], gv[1], gv[2], gcd, gmu]); tape.zero()
                if not (np.isfinite(L) and np.isfinite(g).all()):
                    break
                if L < bloc[0]: bloc = (L, th.copy())
                g = np.clip(g, -50, 50)
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
                lr = np.array([0.08, 0.08, 0.08, 0.06, 0.03])
                th = th - lr * mh / (np.sqrt(vh) + 1e-9)
                th[3] = np.clip(th[3], np.log(2.0), np.log(40.0)); th[4] = np.clip(th[4], 0.0, 1.2)
            if bloc[0] < best[0]:
                best = (bloc[0], bloc[1].copy())
            print(f"  start {si}: best loss {bloc[0]:.3e}")

        L, th = best
        v0 = th[:3]; cd = np.exp(th[3]); mu = th[4]
        fwd(v0, cd, mu)
        rmse = np.sqrt(float(sim.loss.numpy()[0])) * norm / np.sqrt(nf)  # px RMSE
        speed = np.linalg.norm(v0)
        print(f"\nRECOVERED (from the Wan-generated video, via CoTracker + 3D fit):")
        print(f"  launch velocity : ({v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f}) m/s   |v|={speed:.2f} m/s")
        print(f"  restitution-damp: cd={cd:.1f}   friction: mu={mu:.2f}")
        print(f"  fit residual    : {rmse:.1f} px  (over {nf} frames, image {CAM['width']}px wide)")
        rpos = sim.frame_pos()

    _overlay(frames, cen, rpos, cam, trel)
    return 0


def _overlay(frames, cen, rpos, cam, trel):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageSequence
    F = len(frames); H, W, _ = frames[0].shape
    uv, _ = cam.project(rpos)                       # projected sim centre per fit-frame
    puv = np.full((F, 2), np.nan)
    for i in range(len(uv)):
        if trel + i < F:
            puv[trel + i] = uv[i]
    out = REPO / "outputs" / "wan_ball"; out.mkdir(exist_ok=True)
    sc = 0.62
    fig = plt.figure(figsize=(W / 100 * sc, H / 100 * sc), dpi=100); ax = fig.add_axes([0, 0, 1, 1])

    def draw(f):
        ax.clear(); ax.imshow(frames[f]); ax.axis("off")
        ax.scatter([cen[f, 0]], [cen[f, 1]], s=26, c="#ef6a5a", zorder=5)
        if not np.isnan(puv[f, 0]):
            ax.scatter([puv[f, 0]], [puv[f, 1]], s=90, marker="+", c="#57d7e0", zorder=6, linewidths=2.5)
        ax.text(0.02, 0.96, "● tracked   + recovered 3D physics", transform=ax.transAxes,
                color="#f2ece2", fontsize=9, va="top", bbox=dict(boxstyle="round", fc=(0, 0, 0, 0.5), ec="none"))
        return []
    FuncAnimation(fig, draw, frames=F, interval=110).save(out / "recover3d.gif", writer=PillowWriter(fps=9))
    im = Image.open(out / "recover3d.gif")
    fs = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in ImageSequence.Iterator(im)]
    fs[0].save(out / "recover3d.gif", save_all=True, append_images=fs[1:], loop=0, duration=110, optimize=True)
    print(f"wrote {out}/recover3d.gif")


if __name__ == "__main__":
    raise SystemExit(main())
