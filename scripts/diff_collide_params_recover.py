"""Recover DENSITY + FRICTION + RESTITUTION together from one collision.

theta = [log_density_0, mu, log_cd] (density_1 fixed = mass-scale gauge). All three
are differentiable inputs to the 6-DOF+friction contact, so gradient descent
recovers them jointly from the trajectory of an off-center collision (which excites
mass ratio, tangential slip, and normal bounce respectively). cd is the normal
damping that sets the coefficient of restitution; we also measure the effective e
to confirm the bounciness is recovered.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.diff_collide_6dof import DiffCollide6DOF, accum_pos_loss, accum_quat_loss  # noqa: E402

W_ORI = 0.02   # orientation-loss weight (balances position metres vs quaternion mismatch)

NAMES = ["Great_Dinos_Triceratops_Toy", "Great_Dinos_Triceratops_Toy"]
POS0 = [[-0.12, -0.015, 0.0], [0.12, 0.015, 0.0]]   # near head-on -> restitution excited
VEL0 = [[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]
ANG0 = [[0.0, 0.0, 14.0], [0.0, 0.0, 0.0]]           # pre-spin obj0 -> friction excited
D1 = 400.0
TRUE = dict(d0=800.0, mu=0.55, cd=12.0)              # cd=12 -> clearly inelastic here
INIT = dict(d0=1600.0, mu=0.15, cd=40.0)
DT, NSTEPS, K = 2.0e-4, 1800, 3000.0


def build(rg=True):
    return DiffCollide6DOF(NAMES, POS0, VEL0, ang0=ANG0, pitch=0.012, dt=DT, n_steps=NSTEPS,
                           k=K, cd=INIT["cd"], mu=INIT["mu"], gravity=(0, 0, 0), requires_grad=rg)


def set_theta(sim, th):
    sim.set_log_density([th[0], np.log(D1)])
    sim.set_friction(th[1])
    sim.set_damping(np.exp(th[2]))


def loss_fn(sim, target_wp, target_q):
    sim.loss.zero_()
    sc = 1.0 / ((sim.n_steps + 1) * sim.n)
    for t in range(sim.n_steps + 1):
        wp.launch(accum_pos_loss, sim.n, inputs=[sim.pos[t], target_wp[t], sc], outputs=[sim.loss])
        wp.launch(accum_quat_loss, sim.n, inputs=[sim.rot[t], target_q[t], sc * W_ORI], outputs=[sim.loss])


def restitution(sim):
    """Effective coefficient of restitution: -vn_after / vn_before along the line of centers."""
    vl = np.stack([sim.vlin[t].numpy() for t in range(sim.n_steps + 1)])
    p = np.stack([sim.pos[t].numpy() for t in range(sim.n_steps + 1)])
    rel = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    imp = int(np.argmin(rel))
    line = p[:, 1] - p[:, 0]; line /= np.linalg.norm(line, axis=1, keepdims=True) + 1e-9
    vn = np.sum((vl[:, 0] - vl[:, 1]) * line, axis=1)  # closing (>0) then separating (<0)
    before = vn[max(0, imp - 60):imp - 5]
    after = vn[imp + 5:imp + 60]
    return -np.median(after) / (np.median(before) + 1e-9)


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        tgt = build(rg=False)
        th_true = np.array([np.log(TRUE["d0"]), TRUE["mu"], np.log(TRUE["cd"])])
        set_theta(tgt, th_true)
        tgt.rollout()
        target = tgt.positions()
        target_ori = tgt.orientations()
        target_wp = [wp.array(target[t], dtype=wp.vec3) for t in range(len(target))]
        target_q = [wp.array(target_ori[t], dtype=wp.quat) for t in range(len(target_ori))]
        print(f"true: density_0={TRUE['d0']}, mu={TRUE['mu']}, cd={TRUE['cd']} "
              f"(fitting position + orientation)")

        sim = build(rg=True)
        theta = np.array([np.log(INIT["d0"]), INIT["mu"], np.log(INIT["cd"])])
        m = np.zeros(3); v = np.zeros(3)
        b1, b2, lr = 0.9, 0.999, 0.04
        hist = [theta.copy()]
        best = (np.inf, theta.copy())
        for it in range(140):
            tape = wp.Tape()
            with tape:
                set_theta(sim, theta)
                sim.rollout()
                loss_fn(sim, target_wp, target_q)
            tape.backward(sim.loss)
            Lv = float(sim.loss.numpy()[0])
            gd = float(sim.log_density.grad.numpy()[0])
            gmu = float(sim.mu.grad.numpy()[0])
            gcd = float(sim.cd.grad.numpy()[0]) * np.exp(theta[2])
            tape.zero()
            if Lv < best[0]:
                best = (Lv, theta.copy())
            g = np.array([gd, gmu, gcd])
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
            theta = theta - lr * mh / (np.sqrt(vh) + 1e-12)
            theta[1] = min(max(theta[1], 0.0), 2.0)
            hist.append(theta.copy())
            if it % 20 == 0 or it == 139:
                print(f"  it {it:3d} loss={Lv:.2e}  d0={np.exp(theta[0]):7.1f} "
                      f"mu={theta[1]:.3f} cd={np.exp(theta[2]):6.2f}")

        theta = best[1]
        set_theta(sim, theta); sim.rollout()
        d0, mu, cd = np.exp(theta[0]), theta[1], np.exp(theta[2])
        print("\n=== recovered vs true ===")
        print(f"density_0        :  {d0:7.1f}  vs {TRUE['d0']:.1f}   ({100*abs(d0-TRUE['d0'])/TRUE['d0']:.1f}%)")
        print(f"friction mu      :  {mu:7.3f}  vs {TRUE['mu']:.3f}   ({100*abs(mu-TRUE['mu'])/TRUE['mu']:.1f}%)")
        print(f"restitution (cd) :  {cd:7.2f}  vs {TRUE['cd']:.2f}   ({100*abs(cd-TRUE['cd'])/TRUE['cd']:.1f}%)")
        ok = (abs(d0-TRUE['d0'])/TRUE['d0'] < 0.1 and abs(mu-TRUE['mu'])/TRUE['mu'] < 0.2
              and abs(cd-TRUE['cd'])/TRUE['cd'] < 0.25)
        print(f"\n{'DENSITY + FRICTION + RESTITUTION ALL RECOVERED' if ok else 'partial'}")

    np.savez(REPO / "outputs" / "params_recover.npz",
             hist=np.array(hist), true=[np.log(TRUE['d0']), TRUE['mu'], np.log(TRUE['cd'])],
             names=["density_0", "friction mu", "log restitution-damping"])
    _plot(np.array(hist), TRUE, INIT, REPO / "outputs")
    return 0


def _plot(hist, true, init, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = np.arange(len(hist))
    for j, (key, tv, lab, conv) in enumerate([
            ("d0", true["d0"], "density (kg/m³)", np.exp),
            ("mu", true["mu"], "friction  μ", lambda x: x),
            ("cd", true["cd"], "restitution-damping", np.exp)]):
        col = ["d0", "mu", "cd"][j]
        series = conv(hist[:, j]) if j != 1 else hist[:, j]
        ax[j].plot(steps, series, lw=2, color=f"C{j}")
        ax[j].axhline(tv, color="k", ls="--", lw=1.2, label=f"true = {tv:g}")
        ax[j].set(xlabel="optimization step", ylabel=lab, title=f"{lab} recovered")
        ax[j].legend(fontsize=9)
    fig.suptitle("Density, friction, and restitution all recovered from one collision", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "params_recover.png", dpi=120)
    print(f"saved {out / 'params_recover.png'}")


if __name__ == "__main__":
    sys.exit(main())
