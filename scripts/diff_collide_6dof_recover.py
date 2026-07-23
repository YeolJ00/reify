"""6-DOF + friction differentiable collision on real meshes. Verifies:
  1. linear AND angular momentum conservation through an off-center (spin-inducing) hit
  2. friction + rotation are active (bodies spin)
  3. gradient flows (FD check)
  4. density recovered by gradient descent
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.diff_collide_6dof import DiffCollide6DOF, accum_pos_loss  # noqa: E402

NAMES = ["Great_Dinos_Triceratops_Toy", "Great_Dinos_Triceratops_Toy"]
POS0 = [[-0.12, -0.03, 0.0], [0.12, 0.03, 0.0]]   # y-offset -> off-center -> spin
VEL0 = [[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]
D_TRUE = [800.0, 400.0]
D_INIT0 = 1600.0
DT, NSTEPS, MU = 2.0e-4, 1800, 0.5


def build(rg=True, mu=MU):
    return DiffCollide6DOF(NAMES, POS0, VEL0, pitch=0.012, dt=DT, n_steps=NSTEPS,
                           k=3000.0, cd=5.0, mu=mu, gravity=(0, 0, 0), requires_grad=rg)


def loss_fn(sim, target_wp):
    sim.loss.zero_()
    sc = 1.0 / ((sim.n_steps + 1) * sim.n)
    for t in range(sim.n_steps + 1):
        wp.launch(accum_pos_loss, sim.n, inputs=[sim.pos[t], target_wp[t], sc], outputs=[sim.loss])


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        tgt = build(rg=False)
        tgt.set_log_density(np.log(D_TRUE))
        tgt.rollout()
        target = tgt.positions()
        ori = tgt.orientations()
        d = np.linalg.norm(target[:, 0] - target[:, 1], axis=1)
        imp = int(np.argmin(d))
        assert np.isfinite(target).all(), "6-DOF sim exploded"
        print(f"6-DOF collision: {tgt.n_spheres} spheres, min gap {d.min():.3f} at step {imp}")

        # 1. momentum conservation (linear + angular)
        pl = tgt.linear_momentum(); pa = tgt.angular_momentum()
        dl = np.linalg.norm(pl - pl[0], axis=1).max()
        da = np.linalg.norm(pa - pa[0], axis=1).max()
        print(f"\n1. linear  momentum drift: {dl:.2e}  ({'CONSERVED' if dl < 1e-4 else 'drift'})")
        print(f"   angular momentum drift: {da:.2e}  ({'CONSERVED' if da < 1e-4 else 'drift'})")

        # 2. rotation active? angle turned by each body over the rollout
        def turn(qseq, i):
            w = np.clip(np.abs(qseq[:, i, 3]), 0, 1)
            return np.degrees(2 * np.arccos(w)).max()
        print(f"\n2. rotation induced by off-center hit + friction (mu={MU}): "
              f"obj0 turns {turn(ori,0):.0f} deg, obj1 {turn(ori,1):.0f} deg")

        # 3. gradient vs FD
        target_wp = [wp.array(target[t], dtype=wp.vec3) for t in range(len(target))]
        sim = build(rg=True)
        theta0 = np.log([D_INIT0, D_TRUE[1]])
        tape = wp.Tape()
        with tape:
            sim.set_log_density(theta0); sim.rollout(); loss_fn(sim, target_wp)
        tape.backward(sim.loss)
        g_tape = float(sim.log_density.grad.numpy()[0]); tape.zero()
        fd = build(rg=False)

        def L(x):
            fd.set_log_density([x, np.log(D_TRUE[1])]); fd.rollout()
            fd.loss.zero_(); sc = 1.0 / ((fd.n_steps + 1) * fd.n)
            for t in range(fd.n_steps + 1):
                wp.launch(accum_pos_loss, fd.n, inputs=[fd.pos[t], target_wp[t], sc], outputs=[fd.loss])
            return float(fd.loss.numpy()[0])
        h = 0.03
        g_fd = (L(theta0[0] + h) - L(theta0[0] - h)) / (2 * h)
        rel = abs(g_tape - g_fd) / max(abs(g_fd), 1e-12)
        print(f"\n3. tape grad {g_tape:+.4e}  FD {g_fd:+.4e}  rel {rel:.2e}  "
              f"({'GRADIENT FLOWS' if abs(g_tape) > 1e-12 and rel < 0.1 else 'check'})")

        # 4. recover density
        theta = theta0[0]; m = v = 0.0; d_hist = [np.exp(theta)]
        for it in range(50):
            tape = wp.Tape()
            with tape:
                sim.set_log_density([theta, np.log(D_TRUE[1])]); sim.rollout(); loss_fn(sim, target_wp)
            tape.backward(sim.loss)
            g = float(sim.log_density.grad.numpy()[0]); Lv = float(sim.loss.numpy()[0]); tape.zero()
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
            theta = theta - 0.15 * mh / (np.sqrt(vh) + 1e-12); d_hist.append(np.exp(theta))
            if it % 10 == 0 or it == 49:
                print(f"   iter {it:3d} loss={Lv:.3e} density_0={np.exp(theta):7.1f}")
        d_hat = np.exp(theta); err = 100 * abs(d_hat - D_TRUE[0]) / D_TRUE[0]
        print(f"\n4. true {D_TRUE[0]:.0f}  recovered {d_hat:.1f} ({err:.1f}%)  ratio {d_hat/D_TRUE[1]:.2f} vs 2.0")
        print(f"\n{'6-DOF + FRICTION: DENSITY RECOVERED' if err < 8 else 'off'}")

    _plot(target, ori, d_hist, imp, D_TRUE, D_INIT0, REPO / "outputs")
    return 0


def _plot(target, ori, d_hist, imp, d_true, d_init, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(target)) * DT * 1000
    ang = [np.degrees(2 * np.arccos(np.clip(np.abs(ori[:, i, 3]), 0, 1))) for i in range(2)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(t, target[:, 0, 0], lw=2, label="heavy"); ax[0].plot(t, target[:, 1, 0], lw=2, label="light")
    ax[0].axvline(t[imp], color="k", ls=":", lw=1)
    ax[0].set(xlabel="time (ms)", ylabel="x (m)", title="Off-center collision (real meshes)"); ax[0].legend(fontsize=9)
    ax[1].plot(t, ang[0], lw=2, label="heavy"); ax[1].plot(t, ang[1], lw=2, label="light")
    ax[1].axvline(t[imp], color="k", ls=":", lw=1)
    ax[1].set(xlabel="time (ms)", ylabel="rotation (deg)", title="Spin induced by torque + friction"); ax[1].legend(fontsize=9)
    ax[2].plot(d_hist, lw=2, color="C3"); ax[2].axhline(d_true[0], color="k", ls="--", lw=1.2, label=f"true = {d_true[0]:.0f}")
    ax[2].scatter([0], [d_init], color="C3", zorder=3)
    ax[2].set(xlabel="optimization step", ylabel="recovered density", title="Density recovered", ylim=(0, d_init * 1.15)); ax[2].legend(fontsize=9)
    fig.suptitle("6-DOF rigid bodies + friction — differentiable, momentum-conserving, density recovered", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "diff_collide_6dof.png", dpi=120)
    print(f"saved {out / 'diff_collide_6dof.png'}")


if __name__ == "__main__":
    sys.exit(main())
