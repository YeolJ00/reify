"""Recover the object's material (restitution + friction) from the PHOTOREALISTIC
Blender drop video, via LK tracking + differentiable sim + differentiable camera.
Density is a gauge under gravity (M5), so we recover the contact material.
Produces an overlay video: tracks (red) vs recovered simulation (cyan). (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.diff_collide_6dof import _quat_mat  # noqa: E402
from src.sim.diff_drop import DiffDrop  # noqa: E402
from src.track.lk import track_video  # noqa: E402

DROP = REPO / "outputs" / __import__("os").environ.get("SCENE", "drop")


@wp.kernel
def project_drop_loss(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                      pt_local: wp.array(dtype=wp.vec3), target_uv: wp.array(dtype=wp.vec2),
                      valid: wp.array(dtype=float), Rc: wp.mat33, tc: wp.vec3,
                      fx: float, fy: float, cx: float, cy: float,
                      scale: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    world = pos[0] + wp.quat_rotate(rot[0], pt_local[i])
    pc = Rc * world + tc
    z = wp.max(pc[2], 1.0e-6)
    du = fx * pc[0] / z + cx - target_uv[i][0]
    dv = fy * pc[1] / z + cy - target_uv[i][1]
    wp.atomic_add(loss, 0, scale * valid[i] * (du * du + dv * dv))


def main():
    cfg = json.loads((DROP / "scene.json").read_text())
    cam = Camera(cfg["camera"]); true = cfg["true"]; stride = cfg["stride"]
    poses = np.load(DROP / "poses.npz"); pos_t, quat_t = poses["pos"], poses["quat"]
    frames = np.stack([np.asarray(Image.open(DROP / "frames" / f"f{f:03d}.png").convert("RGB"))
                       for f in range(cfg["n_frames"])])
    V = np.array([[float(x) for x in l.split()[1:4]] for l in (DROP / "object.obj").read_text().splitlines()
                  if l.startswith("v ")])

    # --- track ---
    tracks, valid = track_video(frames, max_corners=500, quality=0.006, min_dist=4)
    uv0, _ = cam.project(pos_t[0] + V @ _quat_mat(quat_t[0]).T)
    on_obj = np.array([np.linalg.norm(uv0 - t, axis=1).min() < 8.0 for t in tracks[0]])
    # the object's screen motion — used to reject STATIC background tracks that my
    # frame-0 filter would otherwise keep (they poison the loss)
    com_uv = np.array([cam.project(pos_t[f][None])[0][0] for f in range(len(frames))])
    obj_motion = np.linalg.norm(com_uv[-1] - com_uv[0])

    def tmot(m):
        idx = np.where(valid[:, m])[0]
        return (np.linalg.norm(tracks[idx[-1], m] - tracks[idx[0], m]) if len(idx) > 2 else 0.0)
    mot = np.array([tmot(m) for m in range(tracks.shape[1])])
    keep = on_obj & (valid.sum(0) > 0.35 * len(frames)) & (mot > 0.6 * obj_motion)
    tracks, valid = tracks[:, keep], valid[:, keep]
    tracks = np.nan_to_num(tracks, nan=0.0)   # invalid frames -> 0 (valid mask zeroes them; NaN*0=NaN otherwise)
    M = tracks.shape[1]
    pt_local = np.array([V[np.linalg.norm(uv0 - tracks[0, m], axis=1).argmin()] for m in range(M)], np.float32)
    print(f"tracked {keep.sum()} object points over {len(frames)} frames")
    assert M >= 8, "too few object tracks — move camera closer / add texture"

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        Fobs = len(frames)
        loc = wp.array(pt_local, dtype=wp.vec3)
        tgt = [wp.array(tracks[f].astype(np.float32), dtype=wp.vec2) for f in range(Fobs)]
        val = [wp.array(valid[f].astype(np.float32), dtype=float) for f in range(Fobs)]
        Rc, tc, fx, fy, cx, cy = cam.wp_args()
        norm = cam.width * np.sqrt(Fobs * M)

        def mk(rg):
            return DiffDrop(cfg["name"], cfg["pos0"], cfg["vel0"], cfg["ang0"],
                            density=true["density"], ground_z=cfg["ground_z"], dt=cfg["dt"],
                            n_steps=stride * (Fobs - 1), k=3000.0, cd=4.0, mu=0.2, requires_grad=rg)
        sim = mk(True)

        def fwd(mu, logcd):
            sim.set_friction(mu); sim.set_damping(np.exp(logcd))
            sim.rollout()
            sim.loss.zero_()
            sc = 1.0 / (norm * norm)
            for f in range(Fobs):
                wp.launch(project_drop_loss, M,
                          inputs=[sim.pos[f * stride], sim.rot[f * stride], loc, tgt[f], val[f],
                                  Rc, tc, fx, fy, cx, cy, sc], outputs=[sim.loss])

        LOGCD_LO, LOGCD_HI = np.log(2.5), np.log(35.0)
        th = np.array([0.2, np.log(8.0)])   # neutral init (rises for a thud, falls for a bouncy hit)
        m = np.zeros(2); v = np.zeros(2)
        best = (np.inf, th.copy())
        print(f"recovering (true friction={true['mu']}, restitution-damping cd={true['cd']}) ...")
        for it in range(80):
            tape = wp.Tape()
            with tape:
                fwd(th[0], th[1])
            tape.backward(sim.loss)
            g = np.array([float(sim.mu.grad.numpy()[0]),
                          float(sim.cd.grad.numpy()[0]) * np.exp(th[1])])
            L = float(sim.loss.numpy()[0]); tape.zero()
            if not (np.isfinite(L) and np.isfinite(g).all()):
                th = best[1] + np.array([0.02, 0.05]) * np.random.default_rng(it).standard_normal(2)
                m = np.zeros(2); v = np.zeros(2)   # reset momentum, nudge from best
                continue
            if L < best[0]:
                best = (L, th.copy())
            g = np.clip(g, -300.0, 300.0)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
            th = th - 0.05 * mh / (np.sqrt(vh) + 1e-12)
            th[0] = min(max(th[0], 0.0), 1.4)
            th[1] = min(max(th[1], LOGCD_LO), LOGCD_HI)
            if it % 12 == 0 or it == 79:
                print(f"  it {it:3d} loss={L:.3e} friction={th[0]:.3f} cd={np.exp(th[1]):.2f}")
        th = best[1]
        mu_hat, cd_hat = th[0], np.exp(th[1])
        print(f"\nRECOVERED from the realistic video:")
        print(f"  friction    : {mu_hat:.3f}  vs true {true['mu']:.3f}  ({100*abs(mu_hat-true['mu'])/true['mu']:.0f}%)")
        print(f"  restitution : cd {cd_hat:.2f}  vs true {true['cd']:.2f}  ({100*abs(cd_hat-true['cd'])/true['cd']:.0f}%)")

        sim.set_friction(mu_hat); sim.set_damping(cd_hat); sim.rollout()
        rpos = sim.positions(stride); rori = sim.orientations(stride)

    _overlay(frames, tracks, valid, rpos, rori, pt_local, cam, DROP)
    return 0


def _overlay(frames, tracks, valid, rpos, rori, pt_local, cam, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1])

    def draw(f):
        ax.clear(); ax.imshow(frames[f]); ax.axis("off")
        vv = valid[f]
        world = rpos[f] + pt_local @ _quat_mat(rori[f]).T
        uv, _ = cam.project(world)
        ax.scatter(tracks[f, vv, 0], tracks[f, vv, 1], s=10, c="red", label="observed (tracked)")
        ax.scatter(uv[vv, 0], uv[vv, 1], s=10, marker="+", c="cyan", label="recovered physics")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.6)
        return []

    FuncAnimation(fig, draw, frames=len(frames), interval=70).save(
        out / "drop_recovered.gif", writer=PillowWriter(fps=14))
    plt.close(fig)
    print(f"saved {out / 'drop_recovered.gif'}")


if __name__ == "__main__":
    sys.exit(main())
