"""FULL VIDEO PIPELINE on real scanned assets, end to end:

  real meshes -> simulate a collision (6-DOF + friction) -> RENDER to video ->
  LK point-TRACK the video -> attach tracks to the objects (frame 0) ->
  RECOVER physics (density) by differentiable simulation + differentiable camera
  projection matched to the 2-D tracks -> overlay video (tracks vs recovered sim).

This is steps 2,4,6,7,8,9 of the architecture, on real assets, with the new
differentiable contact under it. Ground truth is known (synthetic-first), so the
recovery is checkable; the output GIF shows it working.

Usage: python scripts/full_pipeline.py
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
from src.track.lk import track_video  # noqa: E402

NAMES = ["Great_Dinos_Triceratops_Toy", "Great_Dinos_Triceratops_Toy"]
POS0 = [[-0.12, -0.015, 0.0], [0.12, 0.015, 0.0]]
VEL0 = [[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]
ANG0 = [[0.0, 0.0, 14.0], [0.0, 0.0, 0.0]]
D_TRUE = [800.0, 400.0]
D_INIT0 = 1600.0
DT, NSTEPS, STRIDE = 2.0e-4, 1800, 36     # 51 observation frames
CAM = {"eye": [0.26, -0.40, 0.30], "target": [0.0, 0.0, 0.0], "fov_deg": 42,
       "width": 520, "height": 400}


@wp.kernel
def project_rigid_loss(
    pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
    body_of: wp.array(dtype=wp.int32), pt_local: wp.array(dtype=wp.vec3),
    target_uv: wp.array(dtype=wp.vec2), valid: wp.array(dtype=float),
    Rc: wp.mat33, tc: wp.vec3, fx: float, fy: float, cx: float, cy: float,
    scale: float, loss: wp.array(dtype=float),
):
    i = wp.tid()
    b = body_of[i]
    world = pos[b] + wp.quat_rotate(rot[b], pt_local[i])
    pc = Rc * world + tc
    z = wp.max(pc[2], 1.0e-6)
    u = fx * pc[0] / z + cx
    v = fy * pc[1] / z + cy
    du = u - target_uv[i][0]
    dv = v - target_uv[i][1]
    wp.atomic_add(loss, 0, scale * valid[i] * (du * du + dv * dv))


def render_frames(pos, ori, V, F, cam, seed=0):
    """Render the collision to RGB frames (textured for LK)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    rng = np.random.default_rng(seed)
    tex = [0.2 + 0.7 * rng.random(len(F)), 0.2 + 0.7 * rng.random(len(F))]  # per-face grays
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
    W, H = cam.width, cam.height
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); axf = fig.add_axes([0, 0, 1, 1])
    frames = []
    for f in range(pos.shape[0]):
        axf.clear(); axf.set_xlim(0, W); axf.set_ylim(H, 0); axf.axis("off"); axf.set_facecolor("white")
        tris, dep, col = [], [], []
        for b in range(2):
            world = pos[f, b] + V @ _quat_mat(ori[f, b]).T
            uv, dd = cam.project(world)
            nrm = np.cross(world[F[:, 1]] - world[F[:, 0]], world[F[:, 2]] - world[F[:, 0]])
            nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
            sh = np.clip(0.4 + 0.6 * np.abs(nrm @ light), 0, 1)
            tz = dd[F].mean(1)
            for kk in range(len(F)):
                tris.append(uv[F[kk]]); dep.append(tz[kk]); col.append(np.array([tex[b][kk]] * 3) * sh[kk])
        o = np.argsort(-np.array(dep))
        axf.add_collection(PolyCollection([tris[k] for k in o], facecolors=[col[k] for k in o], edgecolors="none"))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return np.stack(frames)


def attach(tracks0, pos0, ori0, V, cam):
    """For each frame-0 track, find nearest object vertex -> (body, body-frame point)."""
    proj, bodies, locals_ = [], [], []
    for b in range(2):
        world = pos0[b] + V @ _quat_mat(ori0[b]).T
        uv, dep = cam.project(world)
        proj.append((uv, dep, world))
    body_of = np.zeros(len(tracks0), np.int32)
    pt_local = np.zeros((len(tracks0), 3), np.float32)
    for m, t in enumerate(tracks0):
        best = (1e9, 0, 0)
        for b in range(2):
            uv = proj[b][0]
            dd = np.linalg.norm(uv - t, axis=1)
            j = int(dd.argmin())
            if dd[j] < best[0]:
                best = (dd[j], b, j)
        _, b, j = best
        body_of[m] = b
        # body-frame coordinate of that vertex (undo pose): V[j] is already body-frame
        pt_local[m] = V[proj_vertex_index(proj, b, j)] if False else V[j]
    return body_of, pt_local


def proj_vertex_index(proj, b, j):
    return j


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        # ---- 1. real assets + true simulation ----
        sim = DiffCollide6DOF(NAMES, POS0, VEL0, ang0=ANG0, pitch=0.012, dt=DT, n_steps=NSTEPS,
                              k=3000.0, cd=12.0, mu=0.55, requires_grad=False)
        sim.set_log_density(np.log(D_TRUE)); sim.rollout()
        pos = sim.positions(STRIDE); ori = sim.orientations(STRIDE)   # (Fobs,2,*)
        tm = decimate(load_asset("rigid", NAMES[0]), 500)
        V = (tm.vertices - tm.vertices.mean(0)).astype(np.float64); Fc = tm.faces.astype(np.int32)
        cam = Camera(CAM)

        # ---- 2. render observed video ----
        print("rendering observed video ...")
        frames = render_frames(pos, ori, V, Fc, cam)

        # ---- 3. track ----
        tracks, valid = track_video(frames, max_corners=400, quality=0.008, min_dist=5)
        print(f"tracked {tracks.shape[1]} points, {valid[-1].sum()} survive all {len(frames)} frames")

        # ---- 4. attach to objects (frame 0) ----
        keep = valid[-1]
        tracks = tracks[:, keep]; valid = valid[:, keep]
        body_of, pt_local = attach(tracks[0], pos[0], ori[0], V, cam)
        M = tracks.shape[1]
        print(f"attached {M} tracks: {int((body_of==0).sum())} on obj0, {int((body_of==1).sum())} on obj1")

        # ---- 5. recover density from the 2-D tracks (differentiable sim + projection) ----
        Fobs = len(frames)
        body_wp = wp.array(body_of, dtype=wp.int32)
        loc_wp = wp.array(pt_local, dtype=wp.vec3)
        tgt_uv = [wp.array(tracks[f].astype(np.float32), dtype=wp.vec2) for f in range(Fobs)]
        val_wp = [wp.array(valid[f].astype(np.float32), dtype=float) for f in range(Fobs)]
        Rc, tc, fx, fy, cx, cy = cam.wp_args()
        norm = cam.width * np.sqrt(Fobs * M)

        rec = DiffCollide6DOF(NAMES, POS0, VEL0, ang0=ANG0, pitch=0.012, dt=DT, n_steps=NSTEPS,
                              k=3000.0, cd=12.0, mu=0.55, requires_grad=True)

        def fwd_loss(theta_d0):
            rec.set_log_density([theta_d0, np.log(D_TRUE[1])])
            rec.rollout()
            rec.loss.zero_()
            sc = 1.0 / (norm * norm)
            for f in range(Fobs):
                wp.launch(project_rigid_loss, M,
                          inputs=[rec.pos[f * STRIDE], rec.rot[f * STRIDE], body_wp, loc_wp,
                                  tgt_uv[f], val_wp[f], Rc, tc, fx, fy, cx, cy, sc],
                          outputs=[rec.loss])

        theta = np.log(D_INIT0); mm = vv = 0.0
        print("recovering density from the video ...")
        for it in range(40):
            tape = wp.Tape()
            with tape:
                fwd_loss(theta)
            tape.backward(rec.loss)
            g = float(rec.log_density.grad.numpy()[0]); L = float(rec.loss.numpy()[0]); tape.zero()
            mm = 0.9 * mm + 0.1 * g; vv = 0.999 * vv + 0.001 * g * g
            mh = mm / (1 - 0.9 ** (it + 1)); vh = vv / (1 - 0.999 ** (it + 1))
            theta = theta - 0.15 * mh / (np.sqrt(vh) + 1e-12)
            if it % 8 == 0 or it == 39:
                print(f"  it {it:3d} loss={L:.3e} density_0={np.exp(theta):7.1f}")
        d_hat = np.exp(theta); err = 100 * abs(d_hat - D_TRUE[0]) / D_TRUE[0]
        print(f"\nRECOVERED density_0 = {d_hat:.1f} vs true {D_TRUE[0]:.0f}  ({err:.1f}%)  "
              f"from the VIDEO of real scanned objects")

        # recovered-sim reprojection for the overlay
        rec.set_log_density([theta, np.log(D_TRUE[1])]); rec.rollout()
        rec_pos = rec.positions(STRIDE); rec_ori = rec.orientations(STRIDE)

    _overlay_video(frames, tracks, valid, rec_pos, rec_ori, V, body_of, pt_local, cam, REPO / "outputs")
    return 0


def _overlay_video(frames, tracks, valid, rpos, rori, V, body_of, pt_local, cam, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig = plt.figure(figsize=(cam.width / 100, cam.height / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1])

    def reproj(f):
        pts = np.zeros((len(body_of), 2))
        for m in range(len(body_of)):
            b = body_of[m]
            world = rpos[f, b] + _quat_mat(rori[f, b]) @ pt_local[m]
            uv, _ = cam.project(world[None]); pts[m] = uv[0]
        return pts

    def draw(f):
        ax.clear(); ax.imshow(frames[f]); ax.axis("off")
        vv = valid[f]
        ax.scatter(tracks[f, vv, 0], tracks[f, vv, 1], s=8, c="red", label="tracked (observed)")
        rp = reproj(f)
        ax.scatter(rp[vv, 0], rp[vv, 1], s=8, marker="+", c="cyan", label="recovered simulation")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=60)
    p = out / "full_pipeline.gif"
    anim.save(p, writer=PillowWriter(fps=16)); plt.close(fig)
    print(f"saved {p}  ({p.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    sys.exit(main())
