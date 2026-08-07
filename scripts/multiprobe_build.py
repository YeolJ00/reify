"""Multi-object probe scenes: one simulation and one render per scene, N observations.

Three probes, chosen by the rule in src/sim/tilt_probe.py -- prefer a parameter whose
signature is NOT motion magnitude:

  TILT   friction, from the slip angle. Four flat-based objects, each with a different mu,
         share ONE tilting surface and ONE rollout. Spheres are excluded on measurement:
         a baseball rolls and its slip angle is flat in mu (corr +0.32 against +0.92 for
         the book).
  DROP   restitution. Kept deliberately, as the CONTROL: it is the pairing that is
         confounded with motion magnitude by construction, so it calibrates how much of
         any tilt result is just the confound reappearing.
  DRAPE  cloth stiffness, from the fold shape at REST. Static at the end, so magnitude
         cannot carry it at all.

Scale comes from cropping, not from more renders: every scene is rendered once at full
frame and cropped per object afterwards (src/render/crop.py), so a four-object tilt scene
costs one render and yields four judge queries.

Tilt is produced by rotating GRAVITY and leaving the ground flat. The exported poses are
then rotated back by R_y(theta) about a pivot on the table so the RENDER shows a tilted
table; blender_render_scene.py tilts the table by the same angle.

Run:
  CUDA_VISIBLE_DEVICES=<g> .../envs/warp/bin/python scripts/multiprobe_build.py sim
  LAB=outputs/judge/multi <blender> --background --python scripts/blender_render_scene.py
  .../envs/warp/bin/python scripts/multiprobe_build.py crop
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    DENSITY, FPS, K_CONTACT, PITCH, SUBSTEPS, quat_to_mat, rotz)
from src.sim.tilt_probe import tilt_gravity  # noqa: E402

LAB = REPO / "outputs" / "judge" / "multi"
SRC = REPO / "outputs" / "scene" / "expand"
GZ = 0.706
PIVOT = np.array([-0.05, -0.06, GZ])

# tilt probe: flat-based objects only, spread along -x so they slide into frame
# spacing widened from 0.17 m to 0.30 m after measuring that adjacent crop boxes
# intersected -- the brass pot's crop contained the vase, which makes a per-object
# question ambiguous. Two objects per scene, two scenes, still 1 render per 2 queries.
TILT_OBJECTS = [("book", 0.15, -0.30), ("brass_pot", 0.55, 0.30)]
TILT_OBJECTS_B = [("wooden_bowl", 0.25, -0.30), ("ceramic_vase", 0.80, 0.30)]
# RAMP rather than a fixed incline. At a fixed angle the slide DISTANCE varies enormously
# with mu (measured: 0.02 m to 3.96 m across these four objects), which is a pure
# motion-magnitude difference and would rebuild the confound the probe exists to avoid.
# Ramping the tilt makes the observable the ANGLE AT WHICH EACH OBJECT STARTS TO MOVE --
# a threshold, read per object from one shared render.
TILT_DEG_START, TILT_DEG_END = 4.0, 34.0
TILT_FRAMES = 48

# drop probe (control): same four objects, differing restitution
DROP_OBJECTS = [("book", 1200.0, -0.30), ("brass_pot", 7500.0, 0.30)]
DROP_LIFT = 0.22
DROP_FRAMES = 36


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def prep(cfg, obj):
    """Mesh-derived quantities shared by every rollout of this object."""
    from src.data.assets import decimate, load_asset
    from src.sim.diff_collide_mesh import sphere_cover
    so = cfg["assets"][obj]
    cat, a = so["asset"].split("/")[0], so["asset"].split("/")[-1]
    tm = decimate(load_asset(cat, a), 300).copy()
    tm.apply_scale(so["scale"])
    vm = np.asarray(tm.vertices).mean(0).copy()
    Rz = rotz(so["rot_z"])
    tm.vertices = np.asarray(tm.vertices) @ Rz.T
    ctr, rad = sphere_cover(tm, PITCH * so["scale"])
    return f"{cat}/{a}", so["scale"], vm, Rz, rad, float(ctr[:, 2].min())


def to_poses(P, Q, Rz, vm, rot=None, n=None):
    """Sim trajectory -> Blender poses, optionally rotated into the tilted-table frame."""
    out = []
    for t in range(n if n else len(P)):
        R = quat_to_mat(Q[t, 0]) @ Rz
        loc = P[t, 0] - R @ vm
        if rot is not None:
            loc = PIVOT + rot @ (loc - PIVOT)
            R = rot @ R
        out.append({"loc": [float(x) for x in loc],
                    "mat": [[float(v) for v in r] for r in R]})
    return out


def do_sim():
    import warp as wp
    from src.sim.probe_scene import ProbeScene

    LAB.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "lab.json").read_text())
    wp.init()
    scenes, meta, t0 = {}, {}, time.time()

    with wp.ScopedDevice("cuda:0"):
        # ---- TILT: one rollout per object, all sharing the same incline and clip length
        ramp = np.linspace(TILT_DEG_START, TILT_DEG_END, TILT_FRAMES)
        for _set, _name in ((TILT_OBJECTS, "tilt"), (TILT_OBJECTS_B, "tiltB")):
            tilt_objs, tilt_meta = {}, {}
            for obj, mu, y in _set:
                name, sc, vm, Rz, rad, zmin = prep(cfg, obj)
                pos = [PIVOT[0] - 0.10, y, GZ + rad - zmin + 0.002]
                vel = [0.0, 0.0, 0.0]
                Ps, Qs = [], []
                # step the ramp one frame at a time, carrying state forward
                for f in range(TILT_FRAMES):
                    g = tilt_gravity(np.deg2rad(ramp[f]))
                    s = ProbeScene([name], [pos], [vel], densities=(DENSITY,), ground_z=GZ,
                                   dt=1.0 / (FPS * SUBSTEPS), n_steps=SUBSTEPS,
                                   k=K_CONTACT, cd=3000.0, mu=float(mu),
                                   mesh_scale=[sc], pitch=PITCH, gravity=g)
                    s.rollout()
                    P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                    if not np.isfinite(P).all():
                        break
                    Ps.append(P[-1, 0].copy()); Qs.append(Q[-1, 0].copy())
                    vel = [float(v) for v in (P[-1, 0] - P[0, 0]) * FPS]
                    pos = [float(v) for v in P[-1, 0]]
                if len(Ps) < TILT_FRAMES:
                    print(f"  tilt {obj}: unstable, skipped"); continue
                P = np.array(Ps)[:, None, :]; Q = np.array(Qs)[:, None, :]
                d = np.linalg.norm(P[:, 0, :2] - P[0, 0, :2], axis=1)
                onset = next((f for f in range(TILT_FRAMES) if d[f] > 0.012), None)
                tilt_objs[obj] = to_poses(P, Q, Rz, vm,
                                          rot=None, n=TILT_FRAMES)
                # per-frame rotation: each frame uses ITS OWN ramp angle
                tilt_objs[obj] = [
                    {"loc": [float(x) for x in (PIVOT + ry(np.deg2rad(ramp[t]))
                                                @ (np.array(p["loc"]) - PIVOT))],
                     "mat": [[float(v) for v in r]
                             for r in (ry(np.deg2rad(ramp[t])) @ np.array(p["mat"]))]}
                    for t, p in enumerate(tilt_objs[obj])]
                tilt_meta[obj] = {"mu": mu, "slide_m": round(float(d[-1]), 4),
                                  "onset_frame": onset,
                                  "onset_deg": float(ramp[onset]) if onset is not None else None,
                                  "world_pts": P[:, 0, :].tolist(),
                                  "ramp_deg": ramp.tolist()}
                od = f"{ramp[onset]:.1f}d" if onset is not None else "none"
                print(f"  tilt  {obj:<14} mu={mu:.2f}  onset={od:<6} "
                      f"total slide={d[-1]:.3f} m")
            scenes[_name] = {"tilt_deg": ramp.tolist(), "objects": tilt_objs}
            meta[_name] = tilt_meta

        # ---- DROP control: same objects, restitution varied
        drop_objs, drop_meta = {}, {}
        for obj, cd, y in DROP_OBJECTS:
            name, sc, vm, Rz, rad, zmin = prep(cfg, obj)
            c = [PIVOT[0], y, GZ + rad - zmin + DROP_LIFT]
            s = ProbeScene([name], [c], [[0.0, 0.0, 0.0]], densities=(DENSITY,),
                           ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                           n_steps=DROP_FRAMES * SUBSTEPS, k=K_CONTACT, cd=float(cd),
                           mu=0.5, mesh_scale=[sc], pitch=PITCH)
            s.rollout()
            P, Q = s.positions(SUBSTEPS)[:DROP_FRAMES], s.rotations(SUBSTEPS)[:DROP_FRAMES]
            if not np.isfinite(P).all():
                print(f"  drop {obj}: non-finite, skipped"); continue
            drop_objs[obj] = to_poses(P, Q, Rz, vm, n=DROP_FRAMES)
            drop_meta[obj] = {"cd": cd, "world_pts": P[:, 0, :].tolist()}
            print(f"  drop  {obj:<14} cd={cd:.0f}")
        scenes["drop"] = {"tilt_deg": 0.0, "objects": drop_objs}
        meta["drop"] = drop_meta

    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene_meta.json").write_text(json.dumps(meta))
    (LAB / "lab.json").write_text(
        (REPO / "outputs" / "judge" / "g0c" / "lab.json").read_text())
    n = sum(len(v["objects"]) for v in scenes.values())
    print(f"\n{len(scenes)} scenes, {n} objects total, {time.time()-t0:.0f}s")
    print(f"  renders needed: {len(scenes)}   judge queries available: {n}")
    return 0


def do_crop():
    """Crop each rendered scene per object and write one clip per (scene, object)."""
    import imageio.v2 as imageio
    from src.render.camera import Camera
    from src.render.crop import crop_box, crop_clip, occupancy

    cam = Camera(json.loads((LAB / "lab.json").read_text())["camera"])
    meta = json.loads((LAB / "scene_meta.json").read_text())
    scenes = json.loads((LAB / "scene_poses.json").read_text())
    W, H = 544, 448
    out = []
    for key, sc in scenes.items():
        d = LAB / f"sim_{key}"
        ps = sorted(d.glob("f*.png"))
        if not ps:
            print(f"  {key}: not rendered"); continue
        frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
        # full-scene clip, kept for reference
        fw = imageio.get_writer(str(LAB / f"{key}_FULL.mp4"), fps=int(FPS),
                                codec="libx264", quality=8, macro_block_size=1)
        for f in frames:
            fw.append_data(f)
        fw.close()
        # tilt_deg is a per-frame ramp for the tilt scene and a scalar elsewhere; the
        # crop box must use each frame's OWN angle or it will not follow the object.
        td = sc.get("tilt_deg", 0.0)
        for obj in sc["objects"]:
            pts = np.asarray(meta[key][obj]["world_pts"], float)
            seq = td if isinstance(td, list) else [float(td)] * len(pts)
            pts = np.array([PIVOT + ry(np.deg2rad(seq[min(i, len(seq) - 1)])) @ (p - PIVOT)
                            for i, p in enumerate(pts)])[:, None, :]
            box = crop_box(cam, pts, W, H)
            if box is None:
                print(f"  {key}/{obj}: no crop box"); continue
            c = crop_clip(frames, box)
            dst = LAB / f"{key}__{obj}.mp4"
            w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                                   macro_block_size=1)
            for f in c:
                w.append_data(f)
            w.close()
            occ = occupancy(frames, box)
            rec = {"scene": key, "object": obj, "clip": dst.name, "box": box,
                   "crop_px": [box[2] - box[0], box[3] - box[1]],
                   "motion_in_crop": round(occ, 3), **meta[key][obj]}
            rec.pop("world_pts", None)
            out.append(rec)
            print(f"  {key:<6} {obj:<14} box {box[2]-box[0]:>3}x{box[3]-box[1]:<3} "
                  f"motion {occ:>5.2f}")
    (LAB / "crops.json").write_text(json.dumps(out, indent=2))
    print(f"\n{len(out)} per-object clips from {len(scenes)} renders")
    return 0


MODES = {"sim": do_sim, "crop": do_crop}
if __name__ == "__main__":
    raise SystemExit(MODES[sys.argv[1] if len(sys.argv) > 1 else "sim"]())
