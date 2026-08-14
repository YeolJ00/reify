"""CLOTH DRAPE and SOFT PRESS-RELEASE, on real scanned assets.

Both sweep the parameter first and report the dependence, then the reading -- the rule that
came out of `collide`, which measured nothing because its observable was mass-independent.

CLOTH -- drop the towel and read the FOOTPRINT at rest. Stiff cloth resists folding and lands
spread out; limp cloth crumples into a smaller patch. The readout is taken at rest, so it is a
static shape rather than a motion magnitude.

SOFT -- drop the duck and read peak COMPRESSION and recovery. Excitation is identical for every
candidate (same asset, same height), and the bounce probe established that a magnitude
observable is a fine reading once the object and excitation are held fixed: the -0.88 confound
came from comparing across objects, not from magnitudes as such.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import newton
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.data.assets import load_asset                                  # noqa: E402
from src.sim.deformable import (build_cloth_model, build_soft_model,     # noqa: E402
                                cloth_from_mesh, tets_from_mesh)

FPS = 60.0


def simulate(model, n_frames, substeps, bound=3.0):
    """SolverSemiImplicit: VBD returns exactly-zero gradients in newton 1.4.0.

    DIVERGENCE is checked against a spatial bound, not just isfinite. An explicit solver run
    past its stability limit reaches 10^5 metres long before it reaches inf, so an isfinite
    check reports success on a simulation that has already blown up -- which is exactly what
    the first run of this probe did (a towel with a 455 m 'footprint').
    """
    solver = newton.solvers.SolverSemiImplicit(model)
    st0, st1 = model.state(), model.state()
    control = model.control()
    dt = 1.0 / (FPS * substeps)
    out = [st0.particle_q.numpy().copy()]
    c0 = out[0].mean(0)
    for _ in range(n_frames):
        for _ in range(substeps):
            st0.clear_forces()
            contacts = model.collide(st0)
            solver.step(st0, st1, control, contacts, dt)
            st0, st1 = st1, st0
        q = st0.particle_q.numpy()
        if not np.isfinite(q).all() or np.abs(q - c0).max() > bound:
            return None
        out.append(q.copy())
    return np.stack(out)


def cloth_run(tri_ke, tm, n_frames=60, substeps=256):
    m = build_cloth_model(tm, density=180.0, tri_ke=tri_ke, tri_kd=2.0,
                          edge_ke=tri_ke * 1.0e-3, pos=(0.0, 0.0, 0.45))
    Q = simulate(m, n_frames, substeps)
    if Q is None:
        return None
    end = Q[-1]
    r = np.linalg.norm(end[:, :2] - end[:, :2].mean(0), axis=1)
    return {"footprint_m2": float(np.pi * np.percentile(r, 90) ** 2),
            "spread_cm": float(np.percentile(r, 90) * 100),
            "height_cm": float((end[:, 2].max() - end[:, 2].min()) * 100)}


def soft_run(k_mu, V, T, n_frames=45, substeps=256):
    m = build_soft_model(V, T, density=150.0, k_mu=k_mu, k_lambda=k_mu * 2.0,
                         k_damp=1.0, pos=(0.0, 0.0, 0.22))
    Q = simulate(m, n_frames, substeps)
    if Q is None:
        return None
    h = Q[:, :, 2].max(1) - Q[:, :, 2].min(1)      # body height per frame
    h0 = float(h[0])
    lo = int(np.argmin(h))
    return {"h0_cm": h0 * 100, "compression_pct": float((h0 - h[lo]) / h0 * 100),
            "recovery_pct": float((h[-1] - h[lo]) / max(h0 - h[lo], 1e-9) * 100)}


def main():
    out = Path(os.environ.get("LAB", "outputs/scene/deform")); out.mkdir(parents=True, exist_ok=True)
    wp.init()
    res = {"cloth": [], "soft": []}
    with wp.ScopedDevice("cuda:0"):
        tm = cloth_from_mesh(load_asset("cloth", "Provence_Bath_Towel_Royal_Blue"), 1200, 1.0)
        print(f"=== CLOTH DRAPE  (towel, {len(tm.faces)} faces from 101142)")
        print(f"  {'tri_ke':>9}{'spread cm':>11}{'footprint m2':>14}{'height cm':>11}")
        for ke in [20.0, 60.0, 180.0, 540.0, 1620.0]:
            r = cloth_run(ke, tm)
            if r is None:
                print(f"  {ke:>9.0f}   DIVERGED"); continue
            r["tri_ke"] = ke; res["cloth"].append(r)
            print(f"  {ke:>9.0f}{r['spread_cm']:>11.2f}{r['footprint_m2']:>14.4f}"
                  f"{r['height_cm']:>11.2f}")
        c = res["cloth"]
        if len(c) > 2:
            a = np.array([[x["tri_ke"], x["spread_cm"], x["height_cm"]] for x in c])
            print(f"  rho(log tri_ke, spread) = {np.corrcoef(np.log10(a[:,0]), a[:,1])[0,1]:+.3f}"
                  f"   <- the reading")
            print(f"  rho(log tri_ke, height) = {np.corrcoef(np.log10(a[:,0]), a[:,2])[0,1]:+.3f}")

        V, T = tets_from_mesh(load_asset("soft", "rubber_duck_toy"), pitch=0.016, scale=0.62)
        print(f"\n=== SOFT PRESS-RELEASE  (duck, {len(T)} tets from a real scan)")
        print(f"  {'k_mu':>9}{'h0 cm':>8}{'compress %':>12}{'recover %':>11}")
        for km in [1.0e3, 3.0e3, 9.0e3, 2.7e4, 8.1e4]:
            r = soft_run(km, V, T)
            if r is None:
                print(f"  {km:>9.0f}   DIVERGED"); continue
            r["k_mu"] = km; res["soft"].append(r)
            print(f"  {km:>9.0f}{r['h0_cm']:>8.2f}{r['compression_pct']:>12.2f}"
                  f"{r['recovery_pct']:>11.2f}")
        sf = res["soft"]
        if len(sf) > 2:
            a = np.array([[x["k_mu"], x["compression_pct"], x["recovery_pct"]] for x in sf])
            print(f"  rho(log k_mu, compression) = "
                  f"{np.corrcoef(np.log10(a[:,0]), a[:,1])[0,1]:+.3f}   <- the reading")
            print(f"  rho(log k_mu, recovery)    = "
                  f"{np.corrcoef(np.log10(a[:,0]), a[:,2])[0,1]:+.3f}")
    json.dump(res, open(out / "deform.json", "w"), indent=1)
    print(f"\n  wrote {out}/deform.json")


if __name__ == "__main__":
    main()
