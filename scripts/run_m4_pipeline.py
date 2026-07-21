"""M4 stage 1: recover theta from TRACKED MOTION of a rendered rollout we control.

Pipeline (architecture steps 2, 4, 6, 7, 8, 9):
  render target rollout (matplotlib rasterizer, painter-sorted, textured)
  -> chained pyramidal LK point tracks with forward-backward rejection
  -> attach tracks to the cloth via frame-0 barycentric coordinates
  -> LM recovery on 2D pixel residuals through the (differentiable) pinhole
     projection, gauge-fixed on log_mass.

The FD check of the projection kernel gradient runs first (step-7 sanity).

Usage: python scripts/run_m4_pipeline.py [--starts 3] [--fix log_mass] [--iters 25]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.optimize.lm import lm_recover  # noqa: E402
from src.render.camera import Camera, project_bary_points_kernel  # noqa: E402
from src.sim.theta_sim import (  # noqa: E402
    THETA_DIM,
    THETA_NAMES,
    ThetaFlagSim,
    theta_natural,
    theta_vec_from_cfg,
)
from src.track.lk import attach_barycentric, track_video  # noqa: E402
from src.video.render_flag import render_rollout  # noqa: E402


def check_projection_gradient(camera, sim, track_tri, track_bary):
    """FD-verify d(mean u)/d(particle_q) through the projection kernel (step 7)."""
    q = wp.clone(sim.states[0].particle_q)
    q.requires_grad = True
    uv = wp.zeros(len(track_tri), dtype=wp.vec2, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    @wp.kernel
    def mean_u(uv: wp.array(dtype=wp.vec2), n: float, loss: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(loss, 0, uv[tid][0] / n)

    args = camera.wp_args()
    tape = wp.Tape()
    with tape:
        wp.launch(project_bary_points_kernel, dim=len(track_tri),
                  inputs=[q, sim.model.tri_indices, track_tri, track_bary, *args],
                  outputs=[uv])
        wp.launch(mean_u, dim=len(track_tri), inputs=[uv, float(len(track_tri))], outputs=[loss])
    tape.backward(loss)
    g = q.grad.numpy().copy()
    tape.zero()

    # FD on one particle's x coordinate (pick the one with the largest gradient)
    i = int(np.abs(g).sum(axis=1).argmax())
    h = 1e-3
    qn = q.numpy().copy()

    def f(delta):
        qq = qn.copy()
        qq[i, 0] += delta
        q.assign(qq)
        uv.zero_(); loss.zero_()
        wp.launch(project_bary_points_kernel, dim=len(track_tri),
                  inputs=[q, sim.model.tri_indices, track_tri, track_bary, *args],
                  outputs=[uv])
        wp.launch(mean_u, dim=len(track_tri), inputs=[uv, float(len(track_tri))], outputs=[loss])
        return float(loss.numpy()[0])

    fd = (f(h) - f(-h)) / (2 * h)
    q.assign(qn)
    rel = abs(fd - g[i, 0]) / max(abs(fd), abs(g[i, 0]), 1e-12)
    print(f"projection kernel gradient check: tape={g[i, 0]:+.6e} FD={fd:+.6e} rel={rel:.2e}")
    assert rel < 1e-2, "projection kernel gradient mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "flag.yaml"))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--fix", default="log_mass")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg["seed"])
    wp.init()
    assert wp.get_device(cfg["device"]).is_cuda

    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])
    out = REPO / "outputs"
    out.mkdir(exist_ok=True)

    with wp.ScopedDevice(cfg["device"]):
        # --- steps 2+6: target rollout and forward render ---
        print("target rollout + render ...")
        tgt = ThetaFlagSim(cfg, requires_grad=False)
        tgt.set_theta(theta_true)
        tgt.rollout()
        traj = tgt.trajectory()
        tris = tgt.model.tri_indices.numpy()

        camera = Camera(cfg["m4"]["camera"])
        t0 = time.time()
        frames = render_rollout(traj, tris, camera, cfg["m4"]["render"]["texture_seed"])
        print(f"rendered {frames.shape[0]} frames {frames.shape[1]}x{frames.shape[2]} "
              f"in {time.time() - t0:.1f} s")

        # --- step 4: track ---
        tk = cfg["m4"]["track"]
        tracks, valid = track_video(frames, tk["max_corners"], tk["quality"],
                                    tk["min_dist"], tk["fb_max_px"])
        n_feat = tracks.shape[1]
        survive = valid[-1].sum()
        print(f"LK: {n_feat} features, {survive} survive all {frames.shape[0]} frames")

        # attach to cloth in frame 0
        uv0, depth0 = camera.project(traj[0])
        tri_idx, bary, inside = attach_barycentric(tracks[0], uv0, tris, depth0)
        keep = inside & valid[-1]  # on-cloth tracks that survive the horizon
        tracks = tracks[:, keep]
        tri_idx, bary = tri_idx[keep], bary[keep]
        M = keep.sum()
        print(f"on-cloth full-horizon tracks: {M}")
        assert M >= 20, "too few usable tracks"

        track_tri = wp.array(tri_idx, dtype=wp.int32)
        track_bary = wp.array(bary.astype(np.float32), dtype=wp.vec3)

        # --- step 7 sanity: differentiable projection ---
        sim = ThetaFlagSim(cfg, requires_grad=False)
        check_projection_gradient(camera, sim, track_tri, track_bary)

        # --- steps 8+9: LM on 2D pixel residuals ---
        F = traj.shape[0]
        tgt_uv = tracks[1:].astype(np.float64)  # (F-1, M, 2)
        uv_buf = wp.zeros(int(M), dtype=wp.vec2)
        cam_args = camera.wp_args()
        norm = camera.width * np.sqrt((F - 1) * M)

        fixed = [THETA_NAMES.index(n) for n in args.fix.split(",")] if args.fix else []
        free = [i for i in range(THETA_DIM) if i not in fixed]
        theta0[fixed] = theta_true[fixed]

        def residual_fn(th_free):
            th = theta0.copy()
            th[free] = th_free
            sim.set_theta(th)
            sim.rollout()
            res = np.empty((F - 1, M, 2))
            for f, state in enumerate(sim.frame_states()[1:]):
                wp.launch(project_bary_points_kernel, dim=int(M),
                          inputs=[state.particle_q, sim.model.tri_indices,
                                  track_tri, track_bary, *cam_args],
                          outputs=[uv_buf])
                res[f] = uv_buf.numpy() - tgt_uv[f]
            return res.ravel() / norm

        fd_h = np.array([0.2, 0.2, 0.2, 0.1, 0.04, 0.04, 0.04, 0.04])
        start_sigma = np.array([3.0, 2.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.3])
        rng = np.random.default_rng(cfg["seed"])

        runs = []
        t0 = time.time()
        for s in range(args.starts):
            th_start = theta0[free].copy()
            if s > 0:
                th_start = th_start + rng.normal(0, 1, len(free)) * start_sigma[free]
            print(f"--- start {s}/{args.starts}")
            hat_free, hist = lm_recover(residual_fn, th_start, fd_h[free].copy(), iters=args.iters)
            r = residual_fn(hat_free)
            L = float(r @ r) if np.isfinite(r).all() else np.inf
            runs.append({"start": s, "loss": L, "theta_free": hat_free, "hist": hist})
            print(f"--- start {s} final 2D loss {L:.6e}")
        elapsed = time.time() - t0

        runs.sort(key=lambda r: r["loss"] if np.isfinite(r["loss"]) else np.inf)
        print("\nmulti-start summary:")
        for r in runs:
            print(f"  start {r['start']}  final loss {r['loss']:.6e}")
        winner = runs[0]
        theta_hat = theta0.copy()
        theta_hat[free] = winner["theta_free"]

        # recovered rollout for the overlay plot
        sim.set_theta(theta_hat)
        sim.rollout()
        rec_uv_last = None
        for f, state in enumerate(sim.frame_states()[1:]):
            wp.launch(project_bary_points_kernel, dim=int(M),
                      inputs=[state.particle_q, sim.model.tri_indices,
                              track_tri, track_bary, *cam_args],
                      outputs=[uv_buf])
            if f == F - 2:
                rec_uv_last = uv_buf.numpy().copy()

    # --- report ---
    nat_true, nat_hat, nat_init = theta_natural(theta_true), theta_natural(theta_hat), theta_natural(theta0)
    rms_px = np.sqrt(winner["loss"]) * camera.width
    print(f"\nM4 stage 1 result: 2D loss {winner['loss']:.6e} "
          f"(~{rms_px:.2f} px RMS per track point), {elapsed:.0f} s")
    print(f"{'param':>12} {'true':>12} {'init':>12} {'recovered':>12} {'rel err %':>10}")
    fixed_names = set((args.fix or "").split(","))
    for k in nat_true:
        rel = 100 * abs(nat_hat[k] - nat_true[k]) / max(abs(nat_true[k]), 1e-9)
        fixed_mark = " (fixed)" if (k in fixed_names or f"log_{k}" in fixed_names) else ""
        print(f"{k:>12} {nat_true[k]:12.4f} {nat_init[k]:12.4f} {nat_hat[k]:12.4f} {rel:10.2f}{fixed_mark}")

    np.savez(out / "m4_stage1.npz", theta_true=theta_true, theta_hat=theta_hat,
             tracks=tracks, tri_idx=tri_idx, bary=bary, loss=winner["loss"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, f, title in [(axes[0], 0, "frame 0 + features"), (axes[1], F - 1, "last frame + tracks")]:
        ax.imshow(frames[f])
        ax.set(title=title)
        ax.axis("off")
    axes[0].plot(tracks[0, :, 0], tracks[0, :, 1], "r.", ms=3)
    axes[1].plot(tracks[-1, :, 0], tracks[-1, :, 1], "r.", ms=4, label="LK track")
    axes[1].plot(rec_uv_last[:, 0], rec_uv_last[:, 1], "c+", ms=5, label="recovered sim")
    axes[1].legend(loc="lower right", fontsize=8)
    hist_loss = [h["loss"] for h in winner["hist"]]
    axes[2].semilogy(hist_loss)
    axes[2].set(xlabel="LM iteration", ylabel="2D loss", title="winning start: loss")
    fig.tight_layout()
    fig.savefig(out / "m4_stage1.png", dpi=110)
    print(f"saved: {out / 'm4_stage1.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
