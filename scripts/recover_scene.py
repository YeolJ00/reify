"""M6: recover per-object physics in a real multi-asset scene.

theta = 6 params/object x N objects = [log_density, mu, restitution, v0(3)] each.
Recovery is FD-Jacobian multi-start LM (XPBD contact -> no gradients), one start
per GPU in parallel.

Modes:
  --start K --out FILE   one LM start (seed K) -> JSON.
  --aggregate            best-of + Gauss-Newton identifiability. Reports whether
                         the object-object collision makes RELATIVE density
                         observable (vs the single-object gauge from M5).

Observation = per-object 8 bbox-corner markers per frame, concatenated.
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
from src.sim.scene_sim import PER, SUBNAMES, SceneSim, theta_natural, theta_vec_from_cfg  # noqa: E402

FD_H_OBJ = np.array([0.15, 0.05, 0.05, 0.05, 0.05, 0.05])
SIGMA_OBJ = np.array([0.4, 0.2, 0.15, 0.4, 0.4, 0.4])
OUTDIR = REPO / "outputs" / "scene_starts"


def residual_fn_factory(cfg, target):
    sim = SceneSim(cfg)
    F, M = target.shape[0], target.shape[1]
    norm = 0.2 * np.sqrt(F * M)

    def residual(theta):
        sim.set_theta(theta)
        traj = sim.rollout()
        if not np.isfinite(traj).all():
            return np.full(target.size, 1e3)
        return (traj - target).ravel() / norm

    return residual


def gn_identifiability(cfg, theta_true, n_obj):
    sim = SceneSim(cfg)
    fd = np.tile(FD_H_OBJ, n_obj)

    def traj_at(th):
        sim.set_theta(th)
        return sim.rollout().ravel()

    r0 = traj_at(theta_true)
    dim = len(theta_true)
    J = np.zeros((r0.size, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = fd[i]
        J[:, i] = (traj_at(theta_true + e) - traj_at(theta_true - e)) / (2 * fd[i])
    GN = J.T @ J / (r0.size / 3)
    evals, evecs = np.linalg.eigh(GN)
    return {"sens": np.sqrt(np.diag(GN)), "evals": evals, "evecs": evecs,
            "cond": float(evals[-1] / max(evals[0], 1e-30)), "GN": GN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "scene.yaml"))
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg["seed"])
    n_obj = len(cfg["objects"])
    theta_true = theta_vec_from_cfg(cfg["theta"]["true_values"])
    theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])
    dim = len(theta_true)
    fd = np.tile(FD_H_OBJ, n_obj)
    sigma = np.tile(SIGMA_OBJ, n_obj)
    names = [f"o{k}.{SUBNAMES[j]}" for k in range(n_obj) for j in range(PER)]

    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        gen = SceneSim(cfg)
        gen.set_theta(theta_true)
        target = gen.rollout()
        del gen

        if args.start is not None:
            residual = residual_fn_factory(cfg, target)
            rng = np.random.default_rng(2000 + args.start)
            th0 = theta0.copy() if args.start == 0 else theta0 + rng.normal(0, 1, dim) * sigma
            t0 = time.time()
            hat, _ = lm_recover(residual, th0, fd.copy(), iters=args.iters, verbose=True)
            r = residual(hat)
            L = float(r @ r) if np.isfinite(r).all() else np.inf
            rms_mm = np.sqrt(L) * 0.2 * 1000
            Path(args.out).write_text(json.dumps(
                {"start": args.start, "loss": L, "rms_mm": round(float(rms_mm), 3),
                 "theta_hat": hat.tolist(), "seconds": round(time.time() - t0)}))
            print(f"start {args.start}: RMS {rms_mm:.2f} mm ({round(time.time()-t0)}s)")
            return 0

        if args.aggregate:
            runs = [json.loads(p.read_text()) for p in sorted(OUTDIR.glob("start_*.json"))]
            runs.sort(key=lambda r: r["loss"] if np.isfinite(r["loss"]) else np.inf)
            best = runs[0]
            theta_hat = np.array(best["theta_hat"])
            print("multi-start:", [f"s{r['start']}={r['rms_mm']:.1f}mm" for r in runs])

            ident = gn_identifiability(cfg, theta_true, n_obj)
            nt, nh, n0 = (theta_natural(theta_true, n_obj), theta_natural(theta_hat, n_obj),
                          theta_natural(theta0, n_obj))
            print(f"\nbest fit RMS {best['rms_mm']:.2f} mm   cond {ident['cond']:.2e}")
            flat = lambda L: [x for o in L for x in [o["density"], o["mu"], o["restitution"], *o["v0"]]]
            ft, fh, f0 = flat(nt), flat(nh), flat(n0)
            print(f"{'param':>14} {'true':>9} {'init':>9} {'recovered':>10} {'sensitivity':>12}")
            for i, nm in enumerate(names):
                print(f"{nm:>14} {ft[i]:>9.3f} {f0[i]:>9.3f} {fh[i]:>10.3f} {ident['sens'][i]:>12.3e}")

            # density gauge analysis: sensitivity of each density, and of the
            # ratio direction (d0 up, d1 down) vs the scale direction (both up)
            di = [k * PER for k in range(n_obj)]  # log_density indices
            GN = ident["GN"]
            scale_dir = np.zeros(dim); ratio_dir = np.zeros(dim)
            for k, idx in enumerate(di):
                scale_dir[idx] = 1.0
                ratio_dir[idx] = 1.0 if k == 0 else -1.0
            scale_dir /= np.linalg.norm(scale_dir); ratio_dir /= np.linalg.norm(ratio_dir)
            c_scale = float(scale_dir @ GN @ scale_dir)
            c_ratio = float(ratio_dir @ GN @ ratio_dir)
            print(f"\ndensity SCALE-direction curvature (both densities up): {c_scale:.3e}")
            print(f"density RATIO-direction curvature (d0 up / d1 down):    {c_ratio:.3e}")
            print(f"ratio/scale = {c_ratio / max(c_scale,1e-30):.1f}x  -> "
                  f"{'collision makes RELATIVE density observable while overall scale stays a gauge' if c_ratio > 3*c_scale else 'both density directions similar'}")

            out = REPO / "outputs"
            np.savez(out / "scene_recover.npz", theta_true=theta_true, theta_hat=theta_hat,
                     theta0=theta0, sens=ident["sens"], names=names,
                     c_scale=c_scale, c_ratio=c_ratio)
            _plot(cfg, theta_true, theta_hat, theta0, ident, names, out, n_obj)
            return 0

    ap.error("pass --start K --out FILE, or --aggregate")


def _plot(cfg, tt, th, t0, ident, names, out, n_obj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with wp.ScopedDevice(cfg["device"]):
        sim = SceneSim(cfg)
        sim.set_theta(tt); pt = sim.rollout_poses()
        sim.set_theta(th); ph = sim.rollout_poses()
    tax = np.arange(pt.shape[0]) / cfg["sim"]["fps"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i in range(n_obj):
        axes[0].plot(tax, pt[:, i, 0], color=f"C{i}", label=f"obj{i} x true")
        axes[0].plot(tax, ph[:, i, 0], color=f"C{i}", ls="--")
    axes[0].set(xlabel="t (s)", ylabel="x (m)", title="true — recovered --"); axes[0].legend(fontsize=7)
    axes[1].barh(names, ident["sens"], color="C0"); axes[1].set(xscale="log",
        title="per-param marker sensitivity", xlabel="m per unit θ"); axes[1].invert_yaxis()
    axes[1].tick_params(labelsize=7)
    for i in range(n_obj):
        axes[2].plot(pt[:, i, 0], pt[:, i, 2], color=f"C{i}", label=f"obj{i} true")
        axes[2].plot(ph[:, i, 0], ph[:, i, 2], color=f"C{i}", ls="--")
    axes[2].set(xlabel="x (m)", ylabel="z (m)", title="paths (x-z)"); axes[2].legend(fontsize=7)
    fig.suptitle("M6 multi-asset scene recovery")
    fig.tight_layout(); fig.savefig(out / "scene_recover.png", dpi=120)
    print(f"saved {out / 'scene_recover.png'}")


if __name__ == "__main__":
    sys.exit(main())
