"""M5.2: recover a real asset's rigid physics from its drop trajectory.

theta = [log_density, mu, restitution, v0(3), w0(3)]. Recovery is solver-agnostic
FD-Jacobian LM (XPBD gradients are zero through contact — see check_grad_rigid.py).

Modes:
  --start K --out FILE   run ONE LM start (seed K), write theta_hat + loss JSON.
                         (a driver runs several in parallel, one per GPU)
  --aggregate            merge per-start JSONs, pick best, print table + plot,
                         and run the Gauss-Newton identifiability probe at theta_true.

Observation = 8 bbox-corner marker world positions per frame (translation +
orientation). Loss = mean squared marker distance; reported as RMS in mm.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.optimize.lm import lm_recover  # noqa: E402
from src.sim.rigid_sim import THETA_DIM, THETA_NAMES, RigidDropSim, theta_natural, theta_vec_from_cfg  # noqa: E402

FD_H = np.array([0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10])
START_SIGMA = np.array([0.4, 0.2, 0.15, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0])
OUTDIR = REPO / "outputs" / "rigid_starts"


def make_residual(cfg, target):
    sim = RigidDropSim(cfg)
    F, M = target.shape[0], target.shape[1]
    norm = 0.2 * np.sqrt(F * M)  # 0.2 m ~ object scale

    def residual(theta):
        sim.set_theta(theta)
        traj = sim.rollout()
        if not np.isfinite(traj).all():
            return np.full(target.size, 1e3)
        return (traj - target).ravel() / norm

    return residual, sim


def run_one_start(cfg, target, theta0, theta_true, start_k, iters):
    residual, _ = make_residual(cfg, target)
    rng = np.random.default_rng(1000 + start_k)
    th_start = theta0.copy() if start_k == 0 else theta0 + rng.normal(0, 1, THETA_DIM) * START_SIGMA
    t0 = time.time()
    hat, hist = lm_recover(residual, th_start, FD_H.copy(), iters=iters, verbose=True)
    r = residual(hat)
    L = float(r @ r) if np.isfinite(r).all() else np.inf
    return {"start": start_k, "loss": L, "theta_hat": hat.tolist(),
            "theta_start": th_start.tolist(), "seconds": round(time.time() - t0),
            "n_frames": target.shape[0]}


def gn_identifiability(cfg, theta_true):
    """Gauss-Newton J^T J from FD marker-trajectory Jacobian at theta_true."""
    sim = RigidDropSim(cfg)

    def traj_at(th):
        sim.set_theta(th)
        return sim.rollout().ravel()

    r0 = traj_at(theta_true)
    J = np.zeros((r0.size, THETA_DIM))
    for i in range(THETA_DIM):
        e = np.zeros(THETA_DIM); e[i] = FD_H[i]
        J[:, i] = (traj_at(theta_true + e) - traj_at(theta_true - e)) / (2 * FD_H[i])
    GN = J.T @ J / (r0.size / 3)
    evals, evecs = np.linalg.eigh(GN)
    sens = np.sqrt(np.diag(GN))  # per-param RMS marker sensitivity (m per unit param)
    return {"sens": sens, "evals": evals, "evecs": evecs, "cond": float(evals[-1] / max(evals[0], 1e-30))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "drop.yaml"))
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg["seed"])
    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])

    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        # target from theta_true (regenerated identically in every process)
        gen = RigidDropSim(cfg)
        gen.set_theta(theta_true)
        target = gen.rollout()
        del gen

        if args.start is not None:
            res = run_one_start(cfg, target, theta0, theta_true, args.start, args.iters)
            rms_mm = np.sqrt(res["loss"]) * 0.2 * 1000  # undo norm -> mm
            res["rms_mm"] = round(float(rms_mm), 3)
            Path(args.out).write_text(json.dumps(res))
            print(f"start {args.start}: loss {res['loss']:.4e}  RMS {rms_mm:.2f} mm  ({res['seconds']}s)")
            return 0

        if args.aggregate:
            runs = [json.loads(p.read_text()) for p in sorted(OUTDIR.glob("start_*.json"))]
            assert runs, "no per-start results found; run the parallel starts first"
            runs.sort(key=lambda r: r["loss"] if np.isfinite(r["loss"]) else np.inf)
            best = runs[0]
            theta_hat = np.array(best["theta_hat"])

            print("\nmulti-start summary:")
            for r in runs:
                print(f"  start {r['start']}: RMS {r['rms_mm']:.2f} mm  ({r['seconds']}s)")

            ident = gn_identifiability(cfg, theta_true)
            nat_t, nat_h, nat_0 = theta_natural(theta_true), theta_natural(theta_hat), theta_natural(theta0)
            print(f"\nbest fit RMS {best['rms_mm']:.2f} mm")
            print(f"{'param':>12} {'true':>10} {'init':>10} {'recovered':>10} {'sensitivity':>12}")
            flat = lambda d: ([d["density"], d["mu"], d["restitution"], *d["v0"], *d["w0"]])
            ft, fh, f0 = flat(nat_t), flat(nat_h), flat(nat_0)
            for i, name in enumerate(THETA_NAMES):
                nm = name.replace("log_", "")
                print(f"{nm:>12} {ft[i]:>10.3f} {f0[i]:>10.3f} {fh[i]:>10.3f} {ident['sens'][i]:>12.4e}")
            print(f"\nidentifiability cond number: {ident['cond']:.2e}")
            order = np.argsort(ident["sens"])
            print("least observable:", [THETA_NAMES[i].replace('log_','') for i in order[:3]])
            print("most observable: ", [THETA_NAMES[i].replace('log_','') for i in order[-3:][::-1]])

            out = REPO / "outputs"
            np.savez(out / "rigid_recover.npz", theta_true=theta_true, theta_hat=theta_hat,
                     theta0=theta0, sens=ident["sens"], evals=ident["evals"],
                     runs=json.dumps(runs))
            _plot(cfg, theta_true, theta_hat, theta0, ident, out)
            return 0

    ap.error("pass --start K --out FILE, or --aggregate")


def _plot(cfg, theta_true, theta_hat, theta0, ident, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with wp.ScopedDevice(cfg["device"]):
        sim = RigidDropSim(cfg)
        sim.set_theta(theta_true); tt = sim.rollout()
        sim.set_theta(theta_hat); th = sim.rollout()
        sim.set_theta(theta0); t0 = sim.rollout()
    ct, ch, c0 = tt.mean(1), th.mean(1), t0.mean(1)
    tax = np.arange(len(ct)) / cfg["sim"]["fps"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, lbl in enumerate("xyz"):
        axes[0].plot(tax, ct[:, k], color=f"C{k}", label=f"{lbl} true")
        axes[0].plot(tax, ch[:, k], color=f"C{k}", ls="--")
        axes[0].plot(tax, c0[:, k], color=f"C{k}", ls=":", alpha=0.4)
    axes[0].set(xlabel="t (s)", ylabel="COM (m)", title="true — recovered -- init ⋯")
    axes[0].legend(fontsize=7, ncol=3)

    nm = [n.replace("log_", "") for n in THETA_NAMES]
    axes[1].barh(nm, ident["sens"], color="C0")
    axes[1].set(xscale="log", title="per-param marker sensitivity", xlabel="m per unit θ")
    axes[1].invert_yaxis()

    axes[2].plot(ct[:, 0], ct[:, 2], label="true"); axes[2].plot(ch[:, 0], ch[:, 2], "--", label="recovered")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(xlabel="x (m)", ylabel="z (m)", title="side view (x-z)"); axes[2].legend()
    fig.suptitle(f"M5 rigid recovery — {cfg['asset']['name']}")
    fig.tight_layout(); fig.savefig(out / "rigid_recover.png", dpi=120)
    print(f"saved {out / 'rigid_recover.png'}")


if __name__ == "__main__":
    sys.exit(main())
