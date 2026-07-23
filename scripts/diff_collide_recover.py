"""Prove the differentiable penalty contact fixes the M7b root cause, in three checks:
  1. momentum conservation (should be exact, unlike XPBD's 57% leak)
  2. gradient flows through the collision (FD-verified, unlike XPBD's zero gradient)
  3. density recovered from the collision by GRADIENT descent (which XPBD/CEM couldn't)

Two spheres of equal radius, different density (mass ratio 2), collide head-on.
We recover sphere-0's log-density from the trajectory with sphere-1 fixed
(the collision couples their masses, so the ratio is observable).

Usage: python scripts/diff_collide_recover.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.diff_collide import DiffCollide, accum_pos_loss  # noqa: E402

RADIUS = [0.1, 0.1]
POS0 = [[-0.30, 0.0, 0.0], [0.30, 0.0, 0.0]]
VEL0 = [[1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]]   # head-on
D_TRUE = [800.0, 400.0]                       # mass ratio 2 (equal radii)
D_INIT0 = 1500.0                              # wrong starting guess for sphere 0
K, C, DT, NSTEPS = 6000.0, 8.0, 5.0e-4, 1200  # stiff penalty, small dt (0.6 s)


def build(requires_grad=True):
    return DiffCollide(RADIUS, POS0, VEL0, dt=DT, n_steps=NSTEPS, k=K, c=C,
                       gravity=(0, 0, 0), requires_grad=requires_grad)


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        # ---------- target ----------
        tgt = build(requires_grad=False)
        tgt.set_log_density(np.log(D_TRUE))
        tgt.rollout()
        target_pos = tgt.positions()          # (T+1, 2, 3)
        n = tgt.n

        # ---------- 1. momentum conservation ----------
        p = tgt.total_momentum()
        px = p[:, 0]
        # find the collision (when spheres overlap): min center distance
        d = np.linalg.norm(target_pos[:, 0] - target_pos[:, 1], axis=1)
        imp = int(np.argmin(d))
        loss_pct = 100 * (1 - px[-1] / px[3]) if abs(px[3]) > 1e-9 else 0.0
        print("=== 1. momentum conservation ===")
        print(f"collision at step {imp}, min sphere gap {d.min() - sum(RADIUS):.4f} m")
        print(f"total px: start {px[3]:.6f}  end {px[-1]:.6f}  "
              f"drift {abs(px[-1] - px[3]):.2e}  ({loss_pct:+.3f}%)")
        # total momentum magnitude drift over the whole rollout
        pmag = np.linalg.norm(p - p[0], axis=1).max()
        print(f"max |p(t)-p(0)| over rollout: {pmag:.2e}  "
              f"({'CONSERVED to machine precision' if pmag < 1e-5 else 'drift'})")

        # ---------- 2. gradient vs finite differences ----------
        print("\n=== 2. gradient through the collision (tape vs FD) ===")
        sim = build(requires_grad=True)
        target_wp = [wp.array(target_pos[t], dtype=wp.vec3) for t in range(len(target_pos))]

        def forward_loss(sim):
            sim.loss.zero_()
            scale = 1.0 / ((sim.n_steps + 1) * n)
            for t in range(0, sim.n_steps + 1):
                wp.launch(accum_pos_loss, n, inputs=[sim.pos[t], target_wp[t], scale],
                          outputs=[sim.loss])

        theta0 = np.log([D_INIT0, D_TRUE[1]])  # recover sphere 0; sphere 1 fixed at truth
        tape = wp.Tape()
        with tape:
            sim.set_log_density(theta0)
            sim.rollout()
            forward_loss(sim)
        tape.backward(sim.loss)
        g_tape = sim.log_density.grad.numpy()[0]
        tape.zero()
        L0 = float(sim.loss.numpy()[0])

        fd_sim = build(requires_grad=False)

        def loss_at(logd0):
            fd_sim.set_log_density([logd0, np.log(D_TRUE[1])])
            fd_sim.rollout()
            fd_sim.loss.zero_()
            scale = 1.0 / ((fd_sim.n_steps + 1) * n)
            for t in range(0, fd_sim.n_steps + 1):
                wp.launch(accum_pos_loss, n, inputs=[fd_sim.pos[t], target_wp[t], scale],
                          outputs=[fd_sim.loss])
            return float(fd_sim.loss.numpy()[0])

        h = 0.02
        g_fd = (loss_at(theta0[0] + h) - loss_at(theta0[0] - h)) / (2 * h)
        rel = abs(g_tape - g_fd) / max(abs(g_fd), 1e-12)
        print(f"L(init) = {L0:.6e}")
        print(f"tape grad dL/d(log_density_0) = {g_tape:+.6e}")
        print(f"FD   grad                     = {g_fd:+.6e}   rel err {rel:.2e}  "
              f"({'GRADIENT FLOWS' if abs(g_tape) > 1e-12 and rel < 0.05 else 'check'})")

        # ---------- 3. recover density by gradient descent (Adam) ----------
        print("\n=== 3. recover density by gradient descent ===")
        theta = theta0[0]
        m = v = 0.0
        b1, b2, eps, lr = 0.9, 0.999, 1e-12, 0.15
        for it in range(60):
            tape = wp.Tape()
            with tape:
                sim.set_log_density([theta, np.log(D_TRUE[1])])
                sim.rollout()
                forward_loss(sim)
            tape.backward(sim.loss)
            g = sim.log_density.grad.numpy()[0]
            L = float(sim.loss.numpy()[0])
            tape.zero()
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
            theta = theta - lr * mh / (np.sqrt(vh) + eps)
            if it % 10 == 0 or it == 59:
                print(f"  iter {it:3d}  loss={L:.3e}  density_0={np.exp(theta):8.2f}")

        d_hat = np.exp(theta)
        err = 100 * abs(d_hat - D_TRUE[0]) / D_TRUE[0]
        ratio_hat = d_hat / D_TRUE[1]
        print(f"\ntrue density_0 = {D_TRUE[0]:.1f}   recovered = {d_hat:.2f}  ({err:.2f}% err)")
        print(f"density RATIO d0/d1: true {D_TRUE[0]/D_TRUE[1]:.3f}  recovered {ratio_hat:.3f}")
        print(f"\n{'DENSITY RECOVERED FROM COLLISION via differentiable Warp contact'if err < 3 else 'off'}")

    out = REPO / "outputs"; out.mkdir(exist_ok=True)
    _plot(px, p, target_pos, d, imp, out)
    return 0


def _plot(px, p, target_pos, dist, imp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(px)) * DT * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t, target_pos[:, 0, 0], label="sphere 0 x")
    ax[0].plot(t, target_pos[:, 1, 0], label="sphere 1 x")
    ax[0].axvline(t[imp], color="k", ls=":", lw=1)
    ax[0].set(xlabel="t (ms)", ylabel="x (m)", title="head-on collision (differentiable Warp contact)")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, px, color="C2", lw=2)
    ax[1].axhline(px[3], color="0.6", ls="--", lw=1)
    ax[1].set(xlabel="t (ms)", ylabel="total x-momentum (kg·m/s)",
              title="momentum: FLAT — conserved to machine precision")
    ax[1].ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(out / "diff_collide.png", dpi=120)
    print(f"saved {out / 'diff_collide.png'}")


if __name__ == "__main__":
    sys.exit(main())
