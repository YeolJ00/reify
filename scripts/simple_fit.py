"""Recover restitution, friction and mass ratio directly from the tracks.

The whole estimator, in one pass, with no parameter grid and no simulation:

    velocity fits around each event  ->  the parameter  ->  its error bar

Each probe is analytically invertible (see src/motion/observables.py), so the
previous grid search plus hand-weighted objectives plus two-axis confidence score
collapses into a measurement with an interval. An interval that covers the whole
plausible range IS the "not identifiable" answer, so there is no separate
established / unverified / not-established vocabulary to maintain.

The precondition that used to be a pile of thresholds is now one significance
test: a velocity that is not distinguishable from zero means the motion never
happened, so the take carries no information.

    CPU only, instant:  python scripts/simple_fit.py
    add a forward check: VERIFY=1 CUDA_VISIBLE_DEVICES=<g> python scripts/simple_fit.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.motion.observables import (admissible, collide_observables,  # noqa: E402
                                    combine, drop_observables, slide_observables)

LAB = REPO / "outputs" / "scene" / "fulllab"
FPS = 24.0
PROBE = {"drop": ("restitution", "e", "restitution"),
         "slide": ("friction", "mu", "friction"),
         "collide": ("mass ratio", "m_target/m_mover", "mass_ratio")}


def pixel_scale(cam_cfg, ground_z):
    """Pixels per metre along the lane, plus how much it varies across the lane.

    Only friction needs this, and the variation is a systematic we should quote
    rather than hide: the object changes depth as it slides, so no single scale is
    exactly right.
    """
    from src.render.camera import Camera
    cam = Camera(cam_cfg)
    scales = []
    for x in (-0.34, -0.23, -0.12):
        P = np.array([[x - 0.05, -0.02, ground_z + 0.03],
                      [x + 0.05, -0.02, ground_z + 0.03]])
        uv, _ = cam.project(P)
        scales.append(float(np.hypot(*(uv[1] - uv[0]))) / 0.10)
    return float(np.mean(scales)), float((max(scales) - min(scales)) / np.mean(scales))


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    px_per_m, scale_var = pixel_scale(cfg["camera"], cfg["ground_z"])
    print(f"pixel scale along the lane: {px_per_m:.0f} px/m "
          f"(varies {scale_var*100:.0f}% end to end — a systematic on friction only)\n")

    subjects = sorted({e["subject"] for e in cfg["experiments"].values()})
    out = {}
    for subj in subjects:
        print(f"=== {subj} ===")
        out[subj] = {}
        for kind in ("drop", "slide", "collide"):
            key = f"{subj}_{kind}"
            if key not in cfg["experiments"]:
                continue
            name, sym, pkind = PROBE[kind]
            vals, ses, reasons, contra = [], [], [], []
            for t in sorted(LAB.glob(f"trk_{key}_seed*.npz")):
                d = np.load(t)
                if "subject_cen" not in d:
                    continue
                cen = d["subject_cen"]
                sd = t.stem.split("seed")[1]
                if kind == "drop":
                    r = drop_observables(cen)
                elif kind == "slide":
                    r = slide_observables(cen, px_per_m, FPS)
                else:
                    r = (collide_observables(cen, d["partner_cen"])
                         if "partner_cen" in d else {"ok": False, "why": "no partner track"})
                if r.get("ok"):
                    good = admissible(pkind, r["value"])
                    flag = ""
                    if kind == "drop" and not r.get("bounced"):
                        flag = "  (no significant rebound)"
                    if kind == "collide" and r.get("mover_sped_up"):
                        flag = "  (mover SPED UP through impact -> negative mass)"
                    if kind == "slide" and not r.get("decelerating"):
                        flag = "  (accelerating, not decelerating)"
                    if not good:
                        # outside the physically possible range: the take contradicts
                        # the model, so it is reported but never averaged in
                        flag += "  [CONTRADICTS PHYSICS — excluded]"
                        contra.append((sd, r["value"]))
                    else:
                        vals.append(r["value"]); ses.append(r["se"])
                    print(f"    seed {sd:<3s} {sym} = {r['value']:8.3f} +- {r['se']:.3f}{flag}")
                else:
                    reasons.append(r.get("why", "?"))
            c = combine(vals, ses)
            if c is None:
                why = (f"{len(contra)} contradicted physics" if contra else
                       (reasons[0] if reasons else "no takes"))
                print(f"  {name:11s}: NO MEASUREMENT  ({len(reasons)} no-motion, "
                      f"{len(contra)} contradictory; e.g. {why})")
                out[subj][kind] = {"value": None, "n_rejected": len(reasons),
                                   "n_contradictory": len(contra), "reasons": reasons}
                continue
            lo, hi = c["value"] - c["interval"], c["value"] + c["interval"]
            rel = c["interval"] / max(abs(c["value"]), 1e-9)
            if c["single_take"]:
                # one take has no repeatability term, so its interval is a lower
                # bound on the true uncertainty and must not be called usable
                verdict = "single take — uncertainty unknown"
            elif rel > 0.5:
                verdict = "USELESS (interval spans a factor >3)"
            elif rel < 0.25:
                verdict = "usable"
            else:
                verdict = "weak"
            if contra:
                verdict += f"; {len(contra)} of {len(contra)+c['n']} takes contradicted physics"
            print(f"  {name:11s}: {sym} = {c['value']:.3f} +- {c['interval']:.3f}  "
                  f"[{lo:.3f}, {hi:.3f}]  n={c['n']}  {verdict}")
            print(f"                within-take +-{c['within']:.3f}, "
                  f"between-take +-{c['between']:.3f} "
                  f"({'takes disagree' if c['between'] > c['within'] else 'takes agree'})")
            # n_contradictory must be recorded on the SUCCESS path too, or takes that
            # violated physics disappear from the audit whenever the rest still
            # produced a value (the brass pot lost 3 of its 5 collisions this way)
            out[subj][kind] = c | {"verdict": verdict, "n_rejected": len(reasons),
                                   "n_contradictory": len(contra)}
        print()

    (LAB / "simple_fit.json").write_text(json.dumps(out, indent=2, default=float))

    print("=" * 74)
    print(f"{'object':14s} {'restitution':>18s} {'friction':>18s} {'mass ratio':>18s}")
    print("=" * 74)
    for subj in subjects:
        cells = []
        for kind in ("drop", "slide", "collide"):
            r = out[subj].get(kind)
            cells.append("—" if not r or r.get("value") is None
                         else f"{r['value']:.2f}+-{r['interval']:.2f}")
        print(f"{subj:14s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")
    n = sum(1 for s in out.values() for r in s.values()
            if r.get("value") is not None and str(r.get("verdict", "")).startswith("usable"))
    tot = sum(len(s) for s in out.values())
    print(f"\n{n} of {tot} parameters measured with an interval tighter than +-25% "
          f"from more than one take")
    print("\nCAVEAT on friction: deceleration measures ROLLING RESISTANCE for a ball\n"
          "that rolls rather than slides, which is physically a different quantity\n"
          "from the Coulomb mu the simulator takes. Valid for the vase/pot/book,\n"
          "not for the baseball or apple.")
    print(f"wrote {LAB}/simple_fit.json")

    if os.environ.get("VERIFY"):
        verify(cfg, out)
    return 0


def verify(cfg, out):
    """Forward-check: does a rollout at the measured value reproduce the observable?

    The measurement above is analytic, so this is what enforces the project's
    actual constraint — the answer has to be a real simulation, not just a number
    that satisfies momentum bookkeeping.
    """
    import warp as wp

    from src.data.assets import decimate, load_asset
    from src.motion.observables import drop_observables
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene
    from src.lab.staging import K_CONTACT, PITCH, SUBSTEPS, obj_geom

    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]
    print("\n" + "=" * 74)
    print("FORWARD CHECK — simulate at the measured restitution, re-measure it")
    print("=" * 74)
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        for subj, per in out.items():
            r = per.get("drop")
            if not r or r.get("value") is None:
                continue
            e_obs = r["value"]
            e_sim = None
            # contact damping that yields the measured rebound; a short bisection
            # beats a grid because the map cd -> e is monotone
            lo, hi = 1.0, 400.0
            e = cfg["experiments"][f"{subj}_drop"]
            so = cfg["assets"][subj]
            for _ in range(7):
                cd = float(np.sqrt(lo * hi))
                c, mk, ms = obj_geom(so, e["subject_pos"], load_asset, decimate,
                                     sphere_cover, GZ)
                c[2] += e["lift"]
                s = ProbeScene([mk], [list(c)], [[0.0, 0.0, 0.0]], densities=(600.0,),
                               ground_z=GZ, dt=1.0 / (24 * SUBSTEPS),
                               n_steps=49 * SUBSTEPS, k=K_CONTACT, cd=cd, mu=0.4,
                               mesh_scale=[ms], pitch=PITCH)
                s.rollout(); P = s.positions(SUBSTEPS)[:49]
                if not np.isfinite(P).all():
                    break
                uv, _ = cam.project(P[:, 0])
                q = drop_observables(np.stack([uv[:, 0], uv[:, 1]], 1))
                if not q.get("ok"):
                    break
                e_sim = q["value"]
                if e_sim > e_obs:
                    lo = cd
                else:
                    hi = cd
            if e_sim is None:
                print(f"  {subj:14s} could not be simulated")
            else:
                ok = abs(e_sim - e_obs) <= max(r["interval"], 0.02)
                print(f"  {subj:14s} measured e={e_obs:.3f}+-{r['interval']:.3f}  "
                      f"sim reaches e={e_sim:.3f} at cd={cd:.1f}  "
                      f"{'REPRODUCED' if ok else 'CANNOT REPRODUCE'}")


if __name__ == "__main__":
    raise SystemExit(main())
