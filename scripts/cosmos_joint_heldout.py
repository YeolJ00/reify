"""Joint recovery across several generated videos, then a held-out prediction.

THE SYSTEM OF EQUATIONS
Every probe video of the same two objects shares the same MATERIAL, but each has its
own unknown launch velocity (Cosmos chose it, we did not). So we solve one joint
problem:

    shared   theta = (restitution cd, friction mu, mass ratio)
    per-probe nuisance v0_p = (vx, vy)          <- one pair per video
    minimise  sum_p || project(sim(theta, v0_p)) - tracks_p ||^2

Stacking probes this way is what makes the full theta identifiable: no single video
constrains all of it (see the matrix), but together they do.

THE HELD-OUT TEST
A fourth video (a differently-configured collision) is never used for fitting. We
freeze the jointly recovered material, fit ONLY its launch velocity — the initial
condition cannot transfer, the material must — and compare the prediction against
default parameters. Material fit on one set of experiments predicting a different
experiment is the evidence that the recovery is physical.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/cosmos_joint_heldout.py   (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.probe_scene import ProbeScene  # noqa: E402
from src.track.balls import track_balls  # noqa: E402

OUT = REPO / "outputs" / "probes_i2v"
FIT_PROBES = {"drop": 0, "collide": 2}          # 'push' is rejected: non-physical motion
HELDOUT = ("heldout", 2)
FPS, SUB = 24, 40
DEFAULT = {"cd": 30.0, "mu": 0.6, "ratio": 1.0}
# widened priors: the earlier scan railed at the edges, so let the optimum be interior
BOUNDS = {"cd": (1.0, 120.0, True), "mu": (0.0, 0.8, False), "ratio": (0.25, 4.0, True)}


def load(probe, seed):
    fr = np.load(OUT / f"vid_{probe}_seed{seed}.npz")["frames"]
    A, RA, B, RB = track_balls(fr)
    return fr, A, B


def simulate(cfg, probe, v0, th, nf):
    L = cfg["layouts"][probe]
    sc = ProbeScene(["A", "B"], [L[0], L[1]], [[float(v0[0]), float(v0[1]), 0.0], [0, 0, 0]],
                    densities=(600.0, 600.0 * th["ratio"]), ground_z=cfg["ground_z"],
                    dt=1.0 / (FPS * SUB), n_steps=nf * SUB, k=40000.0,
                    cd=th["cd"], mu=th["mu"], ball_radius=cfg["ball_radius"])
    sc.rollout()
    return sc.positions(SUB)


def rms(cfg, cam, probe, v0, th, A, B, nf):
    P = simulate(cfg, probe, v0, th, nf)
    if not np.isfinite(P).all():
        return 1e6
    uvA, _ = cam.project(P[:nf, 0]); uvB, _ = cam.project(P[:nf, 1])
    e = n = 0
    for uv, T in ((uvA, A), (uvB, B)):
        m = ~np.isnan(T[:, 0])
        if m.sum():
            e += float(((uv[m] - T[m]) ** 2).sum()); n += int(m.sum()) * 2
    return float(np.sqrt(e / max(n, 1)))


def unpack(z, n_probes):
    """z = [log cd, mu, log ratio, (vx,vy) per probe] -> theta, list of v0"""
    th = {"cd": float(np.exp(z[0])), "mu": float(np.clip(z[1], *BOUNDS["mu"][:2])),
          "ratio": float(np.exp(z[2]))}
    th["cd"] = float(np.clip(th["cd"], *BOUNDS["cd"][:2]))
    th["ratio"] = float(np.clip(th["ratio"], *BOUNDS["ratio"][:2]))
    return th, [z[3 + 2 * i: 5 + 2 * i] for i in range(n_probes)]


def joint_loss(cfg, cam, z, data):
    th, v0s = unpack(z, len(data))
    tot = 0.0
    for (probe, (A, B, nf)), v0 in zip(data.items(), v0s):
        tot += rms(cfg, cam, probe, v0, th, A, B, nf) ** 2
    return float(np.sqrt(tot / len(data)))


def fit_v0_only(cfg, cam, probe, th, A, B, nf, rng, iters=8, pop=16):
    """CEM over just the launch velocity, with the material frozen."""
    m = np.array([0.6, -0.1]); s = np.array([0.7, 0.5])
    best = (1e9, m.copy())
    for _ in range(iters):
        cand = m + s * rng.standard_normal((pop, 2))
        sc = np.array([rms(cfg, cam, probe, c, th, A, B, nf) for c in cand])
        el = cand[np.argsort(sc)[:max(3, pop // 4)]]
        if sc.min() < best[0]:
            best = (float(sc.min()), cand[int(np.argmin(sc))].copy())
        m, s = el.mean(0), np.maximum(el.std(0), 0.02)
    return best


def main():
    cfg = json.loads((OUT / "scene.json").read_text())
    cam = Camera(cfg["camera"])
    wp.init()
    rng = np.random.default_rng(0)
    with wp.ScopedDevice("cuda:0"):
        data = {}
        for p, s in FIT_PROBES.items():
            fr, A, B = load(p, s); data[p] = (A, B, len(fr))
            print(f"fit probe '{p}' seed{s}: {len(fr)} frames, tracked "
                  f"{np.mean(~np.isnan(A[:,0]))*100:.0f}%/{np.mean(~np.isnan(B[:,0]))*100:.0f}%")

        # ---- joint CEM over shared theta + per-probe launch velocities ----
        n = len(data)
        m = np.concatenate([[np.log(12.0), 0.3, np.log(1.0)], np.tile([0.6, -0.1], n)])
        s = np.concatenate([[1.0, 0.25, 0.7], np.tile([0.6, 0.4], n)])
        best = (1e9, m.copy())
        POP, IT = 28, 11
        print(f"\njoint fit: {3 + 2*n} unknowns (3 shared material + {n}x2 launch velocity)")
        for it in range(IT):
            cand = m + s * rng.standard_normal((POP, len(m)))
            sc = np.array([joint_loss(cfg, cam, c, data) for c in cand])
            order = np.argsort(sc); el = cand[order[:7]]
            if sc[order[0]] < best[0]:
                best = (float(sc[order[0]]), cand[order[0]].copy())
            m, s = el.mean(0), np.maximum(el.std(0), np.array([0.05, 0.02, 0.05] + [0.03] * (2 * n)))
            if it % 3 == 0 or it == IT - 1:
                th, _ = unpack(m, n)
                print(f"  it {it:2d}: joint rms {best[0]:5.1f} px | cd {th['cd']:6.2f} "
                      f"mu {th['mu']:.3f} ratio {th['ratio']:.3f}")
        th_joint, v0s = unpack(best[1], n)
        print(f"\nJOINTLY RECOVERED MATERIAL: restitution cd={th_joint['cd']:.2f}, "
              f"friction mu={th_joint['mu']:.3f}, mass ratio={th_joint['ratio']:.3f}")
        for (p, (A, B, nf)), v0 in zip(data.items(), v0s):
            print(f"   {p:8s} launch v0={np.round(v0,2)}  per-probe rms {rms(cfg,cam,p,v0,th_joint,A,B,nf):.1f} px")

        # ---- held-out prediction ----
        hp, hs = HELDOUT
        frh, Ah, Bh = load(hp, hs); nfh = len(frh)
        print(f"\nHELD-OUT '{hp}' seed{hs} (never fitted): {nfh} frames")
        r_joint, v_joint = fit_v0_only(cfg, cam, hp, th_joint, Ah, Bh, nfh, rng)
        r_def, v_def = fit_v0_only(cfg, cam, hp, dict(DEFAULT), Ah, Bh, nfh, rng)
        print(f"  recovered material : {r_joint:5.1f} px   (launch {np.round(v_joint,2)})")
        print(f"  default   material : {r_def:5.1f} px   (launch {np.round(v_def,2)})   "
              f"-> {r_def/max(r_joint,1e-9):.2f}x worse")

        Pj = simulate(cfg, hp, v_joint, th_joint, nfh)
        Pd = simulate(cfg, hp, v_def, dict(DEFAULT), nfh)

    np.savez(OUT / "joint_heldout.npz", theta=json.dumps(th_joint), v0s=np.array(v0s),
             r_joint=r_joint, r_def=r_def, Pj=Pj, Pd=Pd, Ah=Ah, Bh=Bh,
             joint_rms=best[0], default=json.dumps(DEFAULT))
    print(f"\nwrote {OUT}/joint_heldout.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
