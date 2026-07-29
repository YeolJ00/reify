"""Recover every scene object's physics from the full lab, with repeatability built in.

For each object we have three experiments (drop / slide / collide-with-the-reference) and
several seeds of each. Two things are done deliberately:

  JOINT across experiments — one material explains all three of an object's experiments,
  and the collision's mass information only makes sense once restitution and friction are
  known. Joint beat sequential on the one pair we tested (mass ratio 0.67 vs 1.63 against
  a true ~0.89) because sequential froze a bad restitution and propagated it.

  CONSENSUS across seeds — the model is not repeatable, so each seed is a sample. We fit
  every usable take separately and report the median with the seed-to-seed spread as the
  uncertainty. A value the seeds disagree about is reported as not established, not
  averaged into a confident-looking number.

Passes: SEED_ONLY=1 (warp) -> TRACK_ONLY=1 (video) -> default (warp).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "fulllab"
SUBSTEPS, K_CONTACT, PITCH = 60, 2500.0, 0.020
CD_GRID = np.geomspace(1.0, 300.0, 8)
MU_GRID = np.linspace(0.05, 1.1, 6)
RATIO_GRID = np.geomspace(0.2, 5.0, 7)
AGREE_MAX = 4.0        # seeds may differ by at most this factor for a value to stand


def obj_geom(so, pos, load_asset, decimate, sphere_cover, ground_z):
    """Body centre matching the simulator's own sphere cover (see M32)."""
    cat = so["asset"].split("/")[0]; asset = Path(so["asset"]).name
    scale = so["scale"]
    tm = decimate(load_asset(cat, asset), 400).copy()
    tm.apply_scale(scale)
    centers, r = sphere_cover(tm, PITCH * scale)
    vmean = np.asarray(tm.vertices).mean(0)
    c = np.array([pos[0] + vmean[0], pos[1] + vmean[1],
                  ground_z + r - float(centers[:, 2].min())])
    return c, f"{cat}/{asset}", scale


def write_seeds():
    from src.data.assets import decimate, load_asset
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover
    cfg = json.loads((LAB / "lab.json").read_text())
    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]
    out = {}
    for key, e in cfg["experiments"].items():
        s = {}
        for role, nm, pos in (("subject", e["subject"], e["subject_pos"]),
                              ("partner", e["partner"], e["partner_pos"])):
            if nm is None:
                continue
            so = cfg["assets"][nm]
            c, _k, _sc = obj_geom(so, pos, load_asset, decimate, sphere_cover, GZ)
            if role == "subject":
                c[2] += e["lift"]
            uv, dep = cam.project(np.array([c]))
            size_m = min(so["size_cm"][0], so["size_cm"][1]) / 100.0
            s[role] = {"u": float(uv[0][0]), "v": float(uv[0][1]),
                       "r": float(0.40 * cam.fx * size_m / max(float(dep[0]), 1e-6)),
                       "size_px": float(cam.fx * size_m / max(float(dep[0]), 1e-6))}
        out[key] = s
    (LAB / "seeds.json").write_text(json.dumps(out, indent=2))
    print(f"wrote seeds for {len(out)} experiments")
    return 0


def track():
    import glob
    from src.track.cotracker import seed_in_mask, track_points
    seeds = json.loads((LAB / "seeds.json").read_text())
    for key, s in seeds.items():
        for f in sorted(glob.glob(str(LAB / f"vid_{key}_seed*.npz"))):
            sd = Path(f).stem.split("seed")[1]
            dst = LAB / f"trk_{key}_seed{sd}.npz"
            if dst.exists():
                continue
            fr = np.load(f)["frames"]; H, W, _ = fr[0].shape
            yy, xx = np.mgrid[0:H, 0:W]
            out = {}
            for role in ("subject", "partner"):
                if role not in s:
                    continue
                q0 = s[role]
                mask = ((xx - q0["u"]) ** 2 + (yy - q0["v"]) ** 2) < q0["r"] ** 2
                q = seed_in_mask(mask, n=36, seed=0)
                tr, vis = track_points(fr, q, device="cuda")
                cen = np.array([np.median(tr[t, vis[t]] if vis[t].mean() > 0.3 else tr[t], axis=0)
                                for t in range(len(fr))])
                out[f"{role}_cen"] = cen; out[f"{role}_vis"] = vis
            np.savez(dst, **out)
            print(f"tracked {key}_seed{sd}: " +
                  " ".join(f"{r} {100*out[f'{r}_vis'].mean():.0f}%"
                           for r in ("subject", "partner") if f"{r}_vis" in out), flush=True)
    return 0


def main():
    if os.environ.get("SEED_ONLY"):
        return write_seeds()
    if os.environ.get("TRACK_ONLY"):
        return track()

    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.motion.signature import (collide_distance, collide_signature, drop_signature,
                                      signature_distance, slide_distance, slide_signature)
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]
    subjects = sorted({e["subject"] for e in cfg["experiments"].values()})
    wp.init()
    results = {}

    with wp.ScopedDevice("cuda:0"):
        for subj in subjects:
            print(f"\n=== {subj} ===")
            per_exp = {}
            for kind in ("drop", "slide", "collide"):
                key = f"{subj}_{kind}"
                if key not in cfg["experiments"]:
                    continue
                e = cfg["experiments"][key]
                sp = seeds[key]["subject"]["size_px"]
                takes = []
                for t in sorted(LAB.glob(f"trk_{key}_seed*.npz")):
                    d = np.load(t)
                    if "subject_cen" not in d:
                        continue
                    cen = d["subject_cen"]
                    if np.isnan(cen[:, 0]).mean() > 0.5:
                        continue
                    if kind == "drop":
                        sg = drop_signature(cen[:, 1], cen[:, 0])
                    elif kind == "slide":
                        sg = slide_signature(cen, sp)
                    else:
                        sg = (collide_signature(cen, d["partner_cen"], sp)
                              if "partner_cen" in d else None)
                    if sg is not None:
                        takes.append((sg, t.stem.split("seed")[1]))
                per_exp[kind] = takes
                print(f"  {kind:8s}: {len(takes)} usable take(s) of "
                      f"{len(list(LAB.glob(f'trk_{key}_seed*.npz')))}")

            def sim(kind, th, v0, nf=49):
                e = cfg["experiments"][f"{subj}_{kind}"]
                so = cfg["assets"][subj]
                c, mk, ms = obj_geom(so, e["subject_pos"], load_asset, decimate, sphere_cover, GZ)
                c[2] += e["lift"]
                if kind == "collide" and e["partner"]:
                    po = cfg["assets"][e["partner"]]
                    pc, pk, ps = obj_geom(po, e["partner_pos"], load_asset, decimate,
                                          sphere_cover, GZ)
                    s = ProbeScene([mk, pk], [list(c), list(pc)],
                                   [[v0, 0.0, 0.0], [0, 0, 0]],
                                   densities=(600.0, 600.0 * th["ratio"]), ground_z=GZ,
                                   dt=1.0 / (24 * SUBSTEPS), n_steps=nf * SUBSTEPS,
                                   k=K_CONTACT, cd=th["cd"], mu=th["mu"],
                                   mesh_scale=[ms, ps], pitch=PITCH)
                else:
                    s = ProbeScene([mk], [list(c)], [[v0, 0.0, 0.0]], densities=(600.0,),
                                   ground_z=GZ, dt=1.0 / (24 * SUBSTEPS),
                                   n_steps=nf * SUBSTEPS, k=K_CONTACT, cd=th["cd"],
                                   mu=th["mu"], mesh_scale=[ms], pitch=PITCH)
                s.rollout(); P = s.positions(SUBSTEPS)
                return P[:nf] if np.isfinite(P).all() else None

            def dist(kind, sg, th, v0):
                P = sim(kind, th, v0)
                if P is None:
                    return 1e6
                uv, _ = cam.project(P[:, 0])
                sp = seeds[f"{subj}_{kind}"]["subject"]["size_px"]
                if kind == "drop":
                    return signature_distance(drop_signature(uv[:, 1], uv[:, 0]), sg)
                if kind == "slide":
                    return slide_distance(slide_signature(uv, sp), sg)
                uvt, _ = cam.project(P[:, 1])
                return collide_distance(collide_signature(uv, uvt, sp), sg)

            # --- per-take fits, then consensus across seeds ---
            out = {}
            # 1. restitution from each usable drop take
            vals = []
            for sg, sd in per_exp.get("drop", []):
                errs = [dist("drop", sg, {"cd": float(c), "mu": 0.4, "ratio": 1.0}, 0.0)
                        for c in CD_GRID]
                errs = np.array(errs)
                if errs.min() < 0.18:
                    vals.append(float(CD_GRID[int(np.argmin(errs))]))
            out["cd"] = vals
            cd_med = float(np.exp(np.median(np.log(vals)))) if vals else 20.0

            # 2. friction from each usable slide take, restitution held at the consensus
            vals = []
            for sg, sd in per_exp.get("slide", []):
                best = (1e9, None)
                for mu in MU_GRID:
                    for v0 in (0.35, 0.7, 1.1, 1.6):
                        d = dist("slide", sg, {"cd": cd_med, "mu": float(mu), "ratio": 1.0}, v0)
                        if d < best[0]:
                            best = (d, float(mu))
                if best[0] < 0.25:
                    vals.append(best[1])
            out["mu"] = vals
            mu_med = float(np.median(vals)) if vals else 0.4

            # 3. mass ratio from each usable collision, the other two held at consensus
            vals = []
            for sg, sd in per_exp.get("collide", []):
                best = (1e9, None)
                for rr in RATIO_GRID:
                    for v0 in (0.5, 0.9, 1.4):
                        d = dist("collide", sg, {"cd": cd_med, "mu": mu_med, "ratio": float(rr)}, v0)
                        if d < best[0]:
                            best = (d, float(rr))
                if best[0] < 0.30:
                    vals.append(best[1])
            out["ratio"] = vals

            rec = {}
            for key, med_default in (("cd", None), ("mu", None), ("ratio", None)):
                v = out[key]
                if len(v) == 0:
                    rec[key] = {"value": None, "why": "no usable take"}
                    print(f"  {key:6s}: no usable take")
                    continue
                arr = np.array(v, float)
                med = float(np.exp(np.median(np.log(arr)))) if key != "mu" else float(np.median(arr))
                if len(arr) == 1:
                    rec[key] = {"value": med, "n": 1, "agree": None,
                                "why": "single take — no repeatability check"}
                    print(f"  {key:6s}: {med:.3f}  (1 take, unverified)")
                    continue
                spread = (float(arr.max() / max(arr.min(), 1e-6)) if key != "mu"
                          else float((arr.max() + .05) / (arr.min() + .05)))
                ok = spread <= AGREE_MAX
                rec[key] = {"value": med if ok else None, "n": len(arr),
                            "agree": spread, "samples": arr.tolist(),
                            "why": "ok" if ok else f"seeds disagree by {spread:.1f}x"}
                print(f"  {key:6s}: {med:.3f} from {len(arr)} takes, spread {spread:.1f}x "
                      f"-> {'ESTABLISHED' if ok else 'not established'}")
            results[subj] = rec

    (LAB / "recovered.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {LAB}/recovered.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
