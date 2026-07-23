"""Differentiable mesh contact on REAL scanned assets: momentum, gradient, recovery.
Two triceratops scans (same geometry, densities 800/400) collide head-on; recover
object-0's density by gradient descent through the mesh-SDF contact.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.diff_collide_mesh import DiffMeshCollide, accum_pos_loss  # noqa: E402

NAMES = ["Great_Dinos_Triceratops_Toy", "Great_Dinos_Triceratops_Toy"]
POS0 = [[-0.12, 0.0, 0.0], [0.12, 0.0, 0.0]]
VEL0 = [[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]
D_TRUE = [800.0, 400.0]
D_INIT0 = 1600.0
DT, NSTEPS = 3.0e-4, 1400


def build(rg=True):
    return DiffMeshCollide(NAMES, POS0, VEL0, pitch=0.012, dt=DT, n_steps=NSTEPS,
                           k=3000.0, c=5.0, gravity=(0, 0, 0), requires_grad=rg)


def loss_fn(sim, target_wp):
    sim.loss.zero_()
    scale = 1.0 / ((sim.n_steps + 1) * sim.n)
    for t in range(sim.n_steps + 1):
        wp.launch(accum_pos_loss, sim.n, inputs=[sim.pos[t], target_wp[t], scale], outputs=[sim.loss])


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        tgt = build(rg=False)
        tgt.set_log_density(np.log(D_TRUE))
        tgt.rollout()
        target = tgt.positions()
        d = np.linalg.norm(target[:, 0] - target[:, 1], axis=1)
        imp = int(np.argmin(d))
        print(f"real mesh collision: {tgt.n_spheres} spheres cover the two meshes "
              f"({tgt.n_pairs} contact pairs), min COM gap {d.min():.3f} m at step {imp}")

        # 1. momentum
        p = tgt.total_momentum()
        drift = np.linalg.norm(p - p[0], axis=1).max()
        print(f"\n1. momentum drift over rollout: {drift:.2e}  "
              f"({'CONSERVED (machine precision)' if drift < 1e-4 else 'drift'})")

        # 2. gradient vs FD
        target_wp = [wp.array(target[t], dtype=wp.vec3) for t in range(len(target))]
        sim = build(rg=True)
        theta0 = np.log([D_INIT0, D_TRUE[1]])
        tape = wp.Tape()
        with tape:
            sim.set_log_density(theta0)
            sim.rollout()
            loss_fn(sim, target_wp)
        tape.backward(sim.loss)
        g_tape = float(sim.log_density.grad.numpy()[0])
        tape.zero()
        fd = build(rg=False)

        def L(logd0):
            fd.set_log_density([logd0, np.log(D_TRUE[1])])
            fd.rollout()
            fd.loss.zero_()
            sc = 1.0 / ((fd.n_steps + 1) * fd.n)
            for t in range(fd.n_steps + 1):
                wp.launch(accum_pos_loss, fd.n, inputs=[fd.pos[t], target_wp[t], sc], outputs=[fd.loss])
            return float(fd.loss.numpy()[0])

        h = 0.03
        g_fd = (L(theta0[0] + h) - L(theta0[0] - h)) / (2 * h)
        rel = abs(g_tape - g_fd) / max(abs(g_fd), 1e-12)
        print(f"2. tape grad {g_tape:+.4e}  FD {g_fd:+.4e}  rel {rel:.2e}  "
              f"({'GRADIENT FLOWS' if abs(g_tape) > 1e-12 and rel < 0.1 else 'check'})")

        # 3. recover density by gradient descent
        theta = theta0[0]
        m = v = 0.0
        d_hist = [np.exp(theta)]
        for it in range(50):
            tape = wp.Tape()
            with tape:
                sim.set_log_density([theta, np.log(D_TRUE[1])])
                sim.rollout()
                loss_fn(sim, target_wp)
            tape.backward(sim.loss)
            g = float(sim.log_density.grad.numpy()[0])
            Lv = float(sim.loss.numpy()[0])
            tape.zero()
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
            theta = theta - 0.15 * mh / (np.sqrt(vh) + 1e-12)
            d_hist.append(np.exp(theta))
            if it % 10 == 0 or it == 49:
                print(f"   iter {it:3d} loss={Lv:.3e} density_0={np.exp(theta):7.1f}")
        d_hat = np.exp(theta)
        self_d_hist = d_hist
        err = 100 * abs(d_hat - D_TRUE[0]) / D_TRUE[0]
        print(f"\n3. true density_0 {D_TRUE[0]:.0f}  recovered {d_hat:.1f}  ({err:.1f}%)  "
              f"ratio {d_hat / D_TRUE[1]:.2f} vs true {D_TRUE[0] / D_TRUE[1]:.2f}")
        print(f"\n{'DENSITY RECOVERED FROM REAL-MESH COLLISION' if err < 6 else 'off'}")

    _plot(target, d_hist, imp, D_TRUE, D_INIT0, REPO / "outputs")
    return 0


def _plot(target, d_hist, imp, d_true, d_init, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(target)) * DT * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t, target[:, 0, 0], lw=2, label="heavy object")
    ax[0].plot(t, target[:, 1, 0], lw=2, label="light object")
    ax[0].axvline(t[imp], color="k", ls=":", lw=1)
    ax[0].annotate("collision", (t[imp], target[imp, 0, 0]), fontsize=9,
                   xytext=(t[imp] + 40, target[imp, 0, 0] - 0.05))
    ax[0].set(xlabel="time (ms)", ylabel="position (m)",
              title="Two real scanned objects collide")
    ax[0].legend(fontsize=9)
    ax[1].plot(d_hist, lw=2, color="C3")
    ax[1].axhline(d_true[0], color="k", ls="--", lw=1.2, label=f"true density = {d_true[0]:.0f}")
    ax[1].scatter([0], [d_init], color="C3", zorder=3)
    ax[1].annotate("wrong guess", (0, d_init), fontsize=9, xytext=(5, d_init))
    ax[1].set(xlabel="optimization step", ylabel="recovered density (kg/m³)",
              title="Gradient descent recovers the density", ylim=(0, d_init * 1.15))
    ax[1].legend(fontsize=9, loc="center right")
    fig.suptitle("Differentiable contact on real meshes — density recovered from a collision",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "diff_collide_mesh.png", dpi=120)
    print(f"saved {out / 'diff_collide_mesh.png'}")


if __name__ == "__main__":
    sys.exit(main())
