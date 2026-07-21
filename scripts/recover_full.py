"""M3: recover the full theta vector (8 params) from a synthetic target.

theta = [wind a0, a1, a2, gravity_z, log tri_ke, log tri_kd, log edge_ke, log mass].
The target is generated with theta_true from configs/flag.yaml; optimization starts
from theta_init. The 8-dim tape gradient is FD-verified at theta_init before
optimizing (per CLAUDE.md working rules), then Adam runs on the tape gradient.

Usage: python scripts/recover_full.py [--iters 150] [--lr 0.15] [--skip-fd]
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

from src.motion.loss import TrajectoryLoss  # noqa: E402
from src.sim.theta_sim import (  # noqa: E402
    THETA_DIM,
    THETA_NAMES,
    ThetaFlagSim,
    theta_natural,
    theta_vec_from_cfg,
)

# FD step per component: linear params (wind, gravity) vs log params
FD_H = np.array([0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05])


def tape_loss_and_grad(sim, loss_mod, theta):
    sim.set_theta(theta)
    tape = wp.Tape()
    with tape:
        sim.rollout()
        loss_mod.compute(sim.frame_states())
    tape.backward(loss_mod.loss)
    wp.synchronize()
    L = loss_mod.value()
    g = sim.theta.grad.numpy().astype(np.float64).copy()
    tape.zero()
    return L, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "flag.yaml"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--method", choices=["grad", "cem"], default="grad")
    ap.add_argument("--skip-fd", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="npz from a previous run; start from its theta_hat (hybrid CEM->grad)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg["sim"]["solver"] == "semi_implicit", "tape gradients require semi_implicit"
    np.random.seed(cfg["seed"])

    wp.init()
    assert wp.get_device(cfg["device"]).is_cuda

    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])
    if args.init_from:
        theta0 = np.load(args.init_from)["theta_hat"].astype(np.float64)
        print(f"init from {args.init_from}: theta0 = {np.round(theta0, 4)}")

    with wp.ScopedDevice(cfg["device"]):
        print("generating synthetic target from theta_true ...")
        tgt_sim = ThetaFlagSim(cfg, requires_grad=False)
        tgt_sim.set_theta(theta_true)
        tgt_sim.rollout()
        target = tgt_sim.trajectory()
        assert np.isfinite(target).all()

        sim = ThetaFlagSim(cfg, requires_grad=True)
        loss_mod = TrajectoryLoss(target, requires_grad=True)

        fd_sim = ThetaFlagSim(cfg, requires_grad=False)
        fd_loss = TrajectoryLoss(target, requires_grad=False)

        def loss_at(theta):
            fd_sim.set_theta(theta)
            fd_sim.rollout()
            fd_loss.compute(fd_sim.frame_states())
            return fd_loss.value()

        if args.method == "cem":
            from src.optimize.cem import cem_nd

            # sigma per component: linear params (wind, gravity) vs log params
            sigma0 = np.array([4.0, 2.0, 2.0, 1.5, 0.8, 0.8, 0.8, 0.5])
            t0 = time.time()
            theta_hat_cem, cem_hist = cem_nd(
                loss_at, mu0=theta0, sigma0=sigma0, pop_size=32,
                iters=args.iters if args.iters < 100 else 25, seed=cfg["seed"],
            )
            elapsed = time.time() - t0
            hist_loss = [h["best_loss"] for h in cem_hist]
            hist_theta = [h["mu"] for h in cem_hist]
            best = (loss_at(theta_hat_cem), theta_hat_cem)
            report(cfg, theta_true, theta0, best, hist_loss, hist_theta, elapsed,
                   loss_at, tag="cem")
            return 0

        # ---- FD verification of the 8-dim gradient at theta0 ----
        L0, g = tape_loss_and_grad(sim, loss_mod, theta0)
        print(f"L(theta_init) = {L0:.6e}")
        if not args.skip_fd:
            print(f"{'param':>12} {'tape grad':>14} {'FD central':>14} {'rel err':>10}")
            worst = 0.0
            for i in range(THETA_DIM):
                e = np.zeros(THETA_DIM)
                e[i] = FD_H[i]
                fd = (loss_at(theta0 + e) - loss_at(theta0 - e)) / (2 * FD_H[i])
                rel = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-12)
                worst = max(worst, rel)
                print(f"{THETA_NAMES[i]:>12} {g[i]:+14.6e} {fd:+14.6e} {rel:10.3e}")
            print(f"worst rel err: {worst:.3e}  ({'OK' if worst < 0.05 else 'SUSPECT — check landscape curvature'})")

        # ---- Adam ----
        print(f"\nAdam: {args.iters} iters, lr={args.lr}")
        theta = theta0.copy()
        m = np.zeros(THETA_DIM)
        v = np.zeros(THETA_DIM)
        beta1, beta2, eps = 0.9, 0.999, 1e-12
        hist_loss, hist_theta = [], []
        best = (np.inf, theta.copy())

        t0 = time.time()
        for it in range(args.iters):
            L, g = tape_loss_and_grad(sim, loss_mod, theta)
            if not np.isfinite(L) or not np.isfinite(g).all():
                print(f"  iter {it}: non-finite loss/grad, stopping early")
                break
            if L < best[0]:
                best = (L, theta.copy())
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** (it + 1))
            v_hat = v / (1 - beta2 ** (it + 1))
            theta = theta - args.lr * m_hat / (np.sqrt(v_hat) + eps)
            hist_loss.append(L)
            hist_theta.append(theta.copy())
            if it % 10 == 0 or it == args.iters - 1:
                print(f"  iter {it:4d}  loss={L:.6e}")
        elapsed = time.time() - t0

        report(cfg, theta_true, theta0, best, hist_loss, hist_theta, elapsed,
               loss_at, tag="grad")
    return 0


def report(cfg, theta_true, theta0, best, hist_loss, hist_theta, elapsed, loss_at, tag):
    theta_hat = best[1]
    L_final = loss_at(theta_hat)
    nat_true, nat_hat = theta_natural(theta_true), theta_natural(theta_hat)
    nat_init = theta_natural(theta0)
    print(f"\nfinal loss {L_final:.6e}  ({elapsed:.0f} s)")
    print(f"{'param':>12} {'true':>12} {'init':>12} {'recovered':>12} {'rel err %':>10}")
    for k in nat_true:
        t_, i_, h_ = nat_true[k], nat_init[k], nat_hat[k]
        rel = 100 * abs(h_ - t_) / max(abs(t_), 1e-9)
        print(f"{k:>12} {t_:12.4f} {i_:12.4f} {h_:12.4f} {rel:10.2f}")

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    np.savez(out / f"recover_full_{tag}.npz", theta_true=theta_true, theta0=theta0,
             theta_hat=theta_hat, loss_hist=np.array(hist_loss),
             theta_hist=np.array(hist_theta))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = np.array(hist_theta)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(hist_loss)
    axes[0].set(xlabel="iteration", ylabel="loss", title=f"M3 loss curve ({tag})")
    for i, name in enumerate(THETA_NAMES):
        (line,) = axes[1].plot(th[:, i], label=name)
        axes[1].axhline(theta_true[i], color=line.get_color(), ls=":", lw=0.8)
    axes[1].set(xlabel="iteration", title="theta components (dotted = true)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"recover_full_{tag}.png", dpi=120)
    print(f"saved: {out / f'recover_full_{tag}.png'}")


if __name__ == "__main__":
    sys.exit(main())
