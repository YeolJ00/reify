"""M3: probe identifiability of theta at theta_true.

Because gravity enters the integrator as an acceleration while all internal and
wind forces are scaled by 1/mass, jointly scaling {mass, tri_ke, tri_kd, edge_ke}
(and approximately the wind coeffs) leaves accelerations nearly unchanged — a
candidate null direction. This script measures it:

1. FD Hessian of the trajectory-MSE loss at theta_true (loss-value based, robust
   to the rugged fine-scale landscape), eigendecomposition, condition number.
2. Alignment of the smallest-eigenvalue direction with the joint-scaling direction.
3. A 2D loss slice over (log_mass, log_tri_ke) to visualize the valley.

Usage: python scripts/probe_identifiability.py [--slice-n 17]
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
from src.sim.theta_sim import THETA_DIM, THETA_NAMES, ThetaFlagSim, theta_vec_from_cfg  # noqa: E402

# FD steps: linear params (wind m/s, gravity m/s^2) then log-space params
H = np.array([0.5, 0.5, 0.5, 0.3, 0.15, 0.15, 0.15, 0.15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "flag.yaml"))
    ap.add_argument("--slice-n", type=int, default=17)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg["seed"])
    wp.init()
    assert wp.get_device(cfg["device"]).is_cuda

    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])

    with wp.ScopedDevice(cfg["device"]):
        print("generating target at theta_true ...")
        tgt = ThetaFlagSim(cfg, requires_grad=False)
        tgt.set_theta(theta_true)
        tgt.rollout()
        target = tgt.trajectory()

        sim = ThetaFlagSim(cfg, requires_grad=False)
        lm = TrajectoryLoss(target, requires_grad=False)

        n_evals = 0

        def loss_at(th):
            nonlocal n_evals
            n_evals += 1
            sim.set_theta(th)
            sim.rollout()
            lm.compute(sim.frame_states())
            return lm.value()

        # ---- FD Hessian (loss values; L(theta_true)=0 exactly by construction) ----
        print("computing FD Hessian ...")
        t0 = time.time()
        Hs = np.zeros((THETA_DIM, THETA_DIM))
        L0 = loss_at(theta_true)
        print(f"L(theta_true) = {L0:.3e} (should be ~0)")
        Lp = np.zeros(THETA_DIM)
        Lm = np.zeros(THETA_DIM)
        for i in range(THETA_DIM):
            e = np.zeros(THETA_DIM)
            e[i] = H[i]
            Lp[i] = loss_at(theta_true + e)
            Lm[i] = loss_at(theta_true - e)
            Hs[i, i] = (Lp[i] - 2 * L0 + Lm[i]) / H[i] ** 2
        for i in range(THETA_DIM):
            for j in range(i + 1, THETA_DIM):
                ei = np.zeros(THETA_DIM); ei[i] = H[i]
                ej = np.zeros(THETA_DIM); ej[j] = H[j]
                Lpp = loss_at(theta_true + ei + ej)
                Lmm = loss_at(theta_true - ei - ej)
                # 7-point formula using the axis evals (2 sim calls per pair)
                Hs[i, j] = Hs[j, i] = (Lpp + Lmm + 2 * L0 - Lp[i] - Lm[i] - Lp[j] - Lm[j]) / (
                    2 * H[i] * H[j]
                )
        print(f"Hessian: {n_evals} rollouts, {time.time() - t0:.0f} s")

        evals, evecs = np.linalg.eigh(Hs)
        print("\neigenvalues (ascending):")
        for k in range(THETA_DIM):
            print(f"  lambda_{k} = {evals[k]:+.4e}")
        print(f"condition number |lmax/lmin|: {abs(evals[-1] / evals[0]):.2e}")

        def show_vec(v, label):
            comps = ", ".join(f"{n}={c:+.2f}" for n, c in zip(THETA_NAMES, v) if abs(c) > 0.15)
            print(f"  {label}: {comps}")

        print("\nleast-identifiable directions (smallest |eigenvalue|):")
        order = np.argsort(np.abs(evals))
        for k in order[:3]:
            show_vec(evecs[:, k], f"lambda={evals[k]:+.3e}")
        print("best-identified direction:")
        show_vec(evecs[:, np.argmax(np.abs(evals))], f"lambda={evals[np.argmax(np.abs(evals))]:+.3e}")

        # joint-scaling direction: log_mass with log_ke, log_kd, log_edge_ke (+ wind a0 in ~log units)
        s = np.zeros(THETA_DIM)
        s[4] = s[5] = s[6] = s[7] = 1.0
        s[0] = theta_true[0]  # d(a0)/d(log a0): wind coeff scales ~multiplicatively
        s /= np.linalg.norm(s)
        null_dir = evecs[:, order[0]]
        align = abs(np.dot(s, null_dir))
        print(f"\n|cos(angle)| between smallest-eig direction and joint mass+force scaling: {align:.3f}")
        curv_s = float(s @ Hs @ s)
        curv_max = float(evals[-1])
        print(f"curvature along scaling direction: {curv_s:.3e}  (vs max {curv_max:.3e}, ratio {curv_s / curv_max:.2e})")

        # ---- 2D slice: log_mass vs log_tri_ke ----
        print("\ncomputing 2D loss slice (log_mass vs log_tri_ke) ...")
        n = args.slice_n
        span = 0.6
        dm = np.linspace(-span, span, n)
        dk = np.linspace(-span, span, n)
        Z = np.zeros((n, n))
        t0 = time.time()
        for a, dmi in enumerate(dm):
            for b, dkj in enumerate(dk):
                th = theta_true.copy()
                th[7] += dmi
                th[4] += dkj
                Z[a, b] = loss_at(th)
        print(f"slice: {n * n} rollouts, {time.time() - t0:.0f} s")

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    np.savez(out / "identifiability.npz", hessian=Hs, evals=evals, evecs=evecs,
             slice_dm=dm, slice_dk=dk, slice_loss=Z, theta_true=theta_true, h=H)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(np.log10(np.abs(Hs) + 1e-12), cmap="viridis")
    axes[0].set_xticks(range(THETA_DIM), THETA_NAMES, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(THETA_DIM), THETA_NAMES, fontsize=7)
    axes[0].set_title("log10 |Hessian| at theta_true")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    cs = axes[1].contourf(dk, dm, np.log10(Z + 1e-10), levels=25, cmap="magma")
    axes[1].plot(0, 0, "w*", ms=14, mec="k")
    axes[1].set(xlabel="delta log_tri_ke", ylabel="delta log_mass",
                title="log10 loss slice (valley = trade-off)")
    fig.colorbar(cs, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "identifiability.png", dpi=120)
    print(f"saved: {out / 'identifiability.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
