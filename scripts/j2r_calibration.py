"""J2-real KILL TEST: can the judge read physical plausibility of REAL objects?

Why this replaces the cloth-flag version. The first J2 asked the judge to name the fabric
of an untextured grey sheet. Nothing in that image says "silk" or "tarp" -- the material
was never depicted, only asserted in the prompt, so the test could not have succeeded and
its failure said nothing about the judge. Here the object carries its own identity: a
ceramic vase looks like a ceramic vase, and everybody -- including a VLM -- has a prior
about how one behaves when dropped. Appearance is fixed by the asset; only MOTION varies
with theta. That is the question the project actually needs answered.

Design
------
* Four objects with strong, distinct, common-knowledge bounce priors: a rubber duck
  (soft, bouncy), a baseball (moderate), a ceramic vase (hard, barely bounces), a brass
  pot (heavy metal, dead). Real meshes, real textures, staged on the wooden table with the
  city HDRI -- the same scene, camera and Cycles settings the generated clips use.
* One swept axis: contact damping cd, which sets restitution. Everything else is held
  fixed, INCLUDING density (600 for all objects). Per-object densities are not known and
  inventing them was a documented failure of the previous pipeline; holding it fixed keeps
  a single varying axis so the readout below is unambiguous.
* Restitution is MEASURED from the 3D rollout (peak rebound speed / peak approach speed),
  not assumed from cd. The axis is therefore physical, not a knob setting.

Readouts
--------
1. Per-object: does the score vary with measured e at all?
2. CROSS-OBJECT SIGN FLIP -- the decisive one. If the judge merely rewards motion
   magnitude (which is exactly how the cloth version failed), its correlation with e has
   the SAME SIGN for every object. Reading material plausibility instead requires
   preferring high e for the rubber duck and low e for the brass pot. That needs no
   ground-truth parameter value -- only the uncontroversial claim that a rubber duck
   bounces more than a brass pot.

PASS: rho(score, e) > 0 for rubber duck AND < 0 for brass pot, with the duck-minus-pot
      difference the largest of any object pair.
FAIL: all objects share a sign -> the judge is reading motion, not material. STOP.

Run (three stages, three environments):
  CUDA_VISIBLE_DEVICES=4 .../envs/warp/bin/python scripts/j2r_calibration.py sim
  LAB=outputs/judge/j2r <blender> --background --python scripts/blender_render_sim.py
  CUDA_VISIBLE_DEVICES=4 .../envs/warp/bin/python scripts/j2r_calibration.py mp4
  HF_HOME=... CUDA_VISIBLE_DEVICES=4 .../envs/cosmos/bin/python scripts/j2r_calibration.py judge
  .../envs/warp/bin/python scripts/j2r_calibration.py report
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "judge" / "j2r"
SRC = REPO / "outputs" / "scene" / "expand"

FPS, NF, SUBSTEPS = 24.0, 60, 60          # 2.5 s -> 10 frames at the judge's 4 fps
K_CONTACT, PITCH, DENSITY = 2500.0, 0.020, 600.0
# Drop height and friction are set together, by measurement. At 0.30 m / mu 0.4 the
# objects toppled and rolled clean off the table -- the brass pot travelled 1.37 m, and
# six of twenty cells left the surface. That matters because horizontal travel rises with
# e, so it would re-create exactly the confound that sank the cloth version: a judge
# rewarding gross motion would look like a judge reading material. At 0.15 m / mu 0.9 the
# worst case is 0.48 m and e still spans 0.03..0.76.
LIFT = 0.15
MU = 0.9
# Every object is dropped at ONE central spot rather than its own staging position. Using
# the staging positions put the rubber duck down on the tabletop's far EDGE, where it
# teetered on the rim for the whole clip -- an artefact that would dominate "is this
# plausible?" for every cell and swamp the restitution axis being tested. This spot is
# near the camera target and clearly inside the surface.
DROP_XY = (-0.05, -0.06)

# Hunt-Crossley damping scales with penetration, so useful cd runs ~1e2..1e4. Chosen to
# spread MEASURED e evenly: above ~5000 the mapping folds back on itself (cd 5000 and
# 20000 both landed near e=0.03-0.12), which would put two indistinguishable cells at the
# bottom of the axis and waste a rank.
CDS = [300.0, 600.0, 1200.0, 2500.0, 5000.0]

# The object is the material. Each carries a common-knowledge bounce prior; "bouncy"
# records the direction a correct judge should lean, and is used ONLY at report time.
OBJECTS = {
    "rubber_duck":  {"noun": "yellow rubber duck",  "bouncy": +1},
    "baseball":     {"noun": "baseball",            "bouncy": +1},
    "ceramic_vase": {"noun": "white ceramic vase",  "bouncy": -1},
    "brass_pot":    {"noun": "brass pot",           "bouncy": -1},
}

PARAPHRASES = (
    "Does the {noun} in this video bounce the way a real {noun} would? Answer yes or no.",
    "Watch how it lands. Is this how a real {noun} behaves when dropped? Answer yes or no.",
    "Judging only by the motion, is this physically plausible for a {noun}? "
    "Answer yes or no.",
)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def restitution(z, fps=FPS):
    """e = peak rebound speed / peak approach speed, from the centroid height alone.

    Averaging over a window or using the trajectory minimum both failed on this project
    (a 1-3 frame bounce gets averaged away, and the minimum finds the post-topple resting
    height). Peak speeds are what survived.
    """
    vz = np.gradient(np.asarray(z, float), 1.0 / fps)
    i_dn = int(np.argmin(vz))
    if i_dn >= len(vz) - 2:
        return None, None
    j = int(np.argmax(vz[i_dn:])) + i_dn
    v_dn, v_up = -vz[i_dn], vz[j]
    if v_dn < 0.2 or v_up <= 0:
        return None, None
    return float(v_up / v_dn), float(v_dn)


# ------------------------------------------------------------------ stage: sim

def do_sim():
    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    LAB.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "lab.json").read_text())
    GZ = cfg["ground_z"]
    wp.init()
    out, rows, t0 = {}, [], time.time()

    with wp.ScopedDevice("cuda:0"):
        for obj, meta in OBJECTS.items():
            so = cfg["assets"][obj]
            cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
            tm = decimate(load_asset(cat, asset), 400).copy()
            tm.apply_scale(so["scale"])
            vmean_local = np.asarray(tm.vertices).mean(0).copy()
            Rz = rotz(so["rot_z"])
            tm.vertices = np.asarray(tm.vertices) @ Rz.T
            centers, rad = sphere_cover(tm, PITCH * so["scale"])
            c = np.array([DROP_XY[0], DROP_XY[1],
                          GZ + rad - float(centers[:, 2].min()) + LIFT])

            for cd in CDS:
                key = f"{obj}_cd{int(cd)}"
                s = ProbeScene([f"{cat}/{asset}"], [list(c)], [[0.0, 0.0, 0.0]],
                               densities=(DENSITY,), ground_z=GZ,
                               dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                               k=K_CONTACT, cd=float(cd), mu=MU,
                               mesh_scale=[so["scale"]], pitch=PITCH)
                s.rollout()
                P = s.positions(SUBSTEPS)[:NF]
                Q = s.rotations(SUBSTEPS)[:NF]
                e, v_dn = restitution(P[:, 0, 2])
                ok = e is not None and np.isfinite(P).all()
                # Horizontal travel is the covariate to watch. The cloth kill test failed
                # because the judge tracked gross motion; if score correlates with this
                # more strongly than with e, the same thing is happening again.
                travel = float(np.linalg.norm(P[:, 0, :2] - P[0, 0, :2], axis=1).max())
                rows.append({"object": obj, "cd": float(cd), "key": key, "e": e,
                             "v_impact": v_dn, "travel_m": round(travel, 3),
                             "ok": bool(ok)})
                if not ok:
                    print(f"  {key:<26} no clean bounce -- dropped")
                    continue
                poses = []
                for t in range(len(P)):
                    R = quat_to_mat(Q[t, 0]) @ Rz
                    loc = P[t, 0] - R @ vmean_local
                    poses.append({"loc": [float(x) for x in loc],
                                  "mat": [[float(v) for v in r] for r in R]})
                out[key] = {"subject": obj, "cd": float(cd), "e": e, "poses": poses}
                print(f"  {key:<26} e={e:.3f}  impact={v_dn:.2f} m/s  ({len(poses)} frames)")

    (LAB / "sim_poses.json").write_text(json.dumps(out))
    (LAB / "cells.json").write_text(json.dumps(rows, indent=2))
    (LAB / "lab.json").write_text(json.dumps({"camera": cfg["camera"],
                                              "ground_z": GZ}))
    print(f"\n{len(out)}/{len(rows)} rollouts usable, {time.time()-t0:.0f}s")
    print(f"wrote {LAB}/sim_poses.json")
    for obj in OBJECTS:
        es = [r["e"] for r in rows if r["object"] == obj and r["ok"]]
        if es:
            print(f"  {obj:<14} e spans {min(es):.3f} .. {max(es):.3f}")
    return 0


# ------------------------------------------------------------------ stage: mp4

def do_mp4():
    """Fold the Blender PNG sequences into mp4s the judge can read."""
    import imageio.v2 as imageio

    poses = json.loads((LAB / "sim_poses.json").read_text())
    made = 0
    for key in poses:
        d = LAB / f"sim_{key}"
        pngs = sorted(d.glob("f*.png"))
        if not pngs:
            print(f"  {key}: no frames rendered")
            continue
        dst = LAB / f"{key}.mp4"
        w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                               macro_block_size=1)
        for p in pngs:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        made += 1
        print(f"  {key}: {len(pngs)} frames -> {dst.name}")
    print(f"\n{made} clips written")
    return 0


# ---------------------------------------------------------------- stage: judge

def do_judge():
    from src.judge.cosmos import CosmosJudge

    rows = [r for r in json.loads((LAB / "cells.json").read_text()) if r["ok"]]
    rows = [r for r in rows if (LAB / f"{r['key']}.mp4").exists()]
    j = CosmosJudge()
    t0, n = time.time(), 0
    for r in rows:
        noun = OBJECTS[r["object"]]["noun"]
        vals = []
        for p in PARAPHRASES:
            s, _, _ = j.score_one(LAB / f"{r['key']}.mp4", p.format(noun=noun))
            vals.append(s)
            n += 1
        a = np.asarray(vals, float)
        r["score"] = float(a.mean())
        r["spread"] = float(a.max() - a.min())
        r["per_prompt"] = [float(x) for x in a]
        print(f"  {r['key']:<26} e={r['e']:.3f}  score={r['score']:+.3f} "
              f"(spread {r['spread']:.2f})")
    (LAB / "scored.json").write_text(json.dumps(rows, indent=2))
    dt = time.time() - t0
    print(f"\n{n} judge calls in {dt:.0f}s ({dt/max(n,1):.2f} s/call)")
    print("decode check:", getattr(j, "_checked", None))
    return 0


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------- stage: report

def do_report():
    rows = json.loads((LAB / "scored.json").read_text())
    print(f"\n{'object':<14} {'n':>2} {'e range':>14} {'rho(score,e)':>13} "
          f"{'spread':>7} {'prior':>6}")
    print("-" * 64)
    res = {}
    for obj, meta in OBJECTS.items():
        rs = [r for r in rows if r["object"] == obj]
        if len(rs) < 3:
            print(f"{obj:<14} {len(rs):>2}  too few cells")
            continue
        e = [r["e"] for r in rs]
        s = [r["score"] for r in rs]
        rho = spearman(s, e)
        res[obj] = rho
        print(f"{obj:<14} {len(rs):>2} {min(e):>6.2f}..{max(e):<6.2f} {rho:>+13.3f} "
              f"{np.mean([r['spread'] for r in rs]):>7.2f} "
              f"{'bouncy' if meta['bouncy'] > 0 else 'dead':>6}")

    # The confound check. The cloth version failed because score tracked gross motion; if
    # travel beats e here, the same artefact is back under a new costume.
    allr = [r for r in rows if r["object"] in res]
    r_e = spearman([r["score"] for r in allr], [r["e"] for r in allr])
    r_t = spearman([r["score"] for r in allr], [r["travel_m"] for r in allr])
    print(f"\npooled over all objects:  rho(score, e) = {r_e:+.3f}   "
          f"rho(score, travel) = {r_t:+.3f}")
    if abs(r_t) > abs(r_e):
        print("  WARNING: score tracks horizontal travel more than restitution.")

    print("\nDECISIVE READOUT -- cross-object sign flip")
    duck, pot = res.get("rubber_duck"), res.get("brass_pot")
    if duck is None or pot is None:
        print("  missing objects; cannot decide")
        return 1
    gap = duck - pot
    signs_ok = duck > 0 and pot < 0
    print(f"  rho(rubber_duck) = {duck:+.3f}   (should be > 0: a duck bounces)")
    print(f"  rho(brass_pot)   = {pot:+.3f}   (should be < 0: a pot does not)")
    print(f"  difference       = {gap:+.3f}")
    all_same = len({np.sign(v) for v in res.values()}) == 1
    if all_same:
        print("\n  Every object shares one sign -> the judge is ranking MOTION, not "
              "material.")
    verdict = "PASS" if (signs_ok and gap > 0.5) else "FAIL"
    print(f"\nVERDICT: {verdict}")
    (LAB / "verdict.json").write_text(json.dumps(
        {"rho": res, "gap": gap, "verdict": verdict}, indent=2))
    return 0


MODES = {"sim": do_sim, "mp4": do_mp4, "judge": do_judge, "report": do_report}

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "report"
    if m not in MODES:
        print(f"usage: {sys.argv[0]} [{'|'.join(MODES)}]")
        raise SystemExit(2)
    raise SystemExit(MODES[m]())
