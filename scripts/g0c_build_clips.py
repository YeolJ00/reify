"""Build a motion-dense Cycles sweep for the G0 rerun, and guard every clip.

Three defects are fixed together here:

* Drop height goes back to 0.30 m. It was lowered to 0.15 m to stop objects rolling off
  the table, which starved the judge of motion (1-2 of its 9 frame deltas contained any).
  Trimming to the event window removes the roll-away tail, so the height can go back up --
  the trim solves the problem the lowering was meant to solve, without the side effect.
* Only the event window is rendered. Cheaper AND denser: no frames are spent on an object
  sitting still, and the judge's fixed 12 frames now span drop, bounce and settle.
* CYCLES, not EEVEE. EEVEE Next defaults to use_raytracing=False, which renders the brass
  pot as a dark matte lump instead of polished metal -- a rule-2 violation (fixed
  target-material appearance) on precisely the object whose preference flipped hardest.
  The EEVEE swap was validated by correlating scores and never by looking at it.

Grid: 6 damping x 5 friction = 30 theta per object, two objects. That is 435 candidate
pairs per object, enough for the powered pairwise test that G0 v1 lacked at n=5.

Run:
  CUDA_VISIBLE_DEVICES=<g> .../envs/warp/bin/python scripts/g0c_build_clips.py sim
  LAB=outputs/judge/g0c <blender> --background --python scripts/blender_render_sim.py
  .../envs/warp/bin/python scripts/g0c_build_clips.py encode
"""
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    DENSITY, DROP_XY, FPS, K_CONTACT, PITCH, SUBSTEPS,
    quat_to_mat, restitution, rotz)
from src.render.motion_budget import event_window, check  # noqa: E402

LAB = REPO / "outputs" / "judge" / "g0c"
SRC = REPO / "outputs" / "scene" / "expand"
OBJS = ("rubber_duck", "brass_pot")
LIFT = 0.30
NF = 96                      # simulate generously; only the event window is rendered
CDS = [300.0, 600.0, 1200.0, 2500.0, 5000.0, 10000.0]
MUS = [0.15, 0.30, 0.50, 0.80, 1.20]
MAX_WIN = 40                 # cap the rendered window (~1.7 s) so cost stays bounded


def do_sim():
    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    LAB.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "lab.json").read_text())
    GZ = cfg["ground_z"]
    wp.init()
    cache, poses_out, rows = {}, {}, []
    t0 = time.time()
    with wp.ScopedDevice("cuda:0"):
        for obj in OBJS:
            so = cfg["assets"][obj]
            cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
            tm = decimate(load_asset(cat, asset), 400).copy()
            tm.apply_scale(so["scale"])
            vmean = np.asarray(tm.vertices).mean(0).copy()
            Rz = rotz(so["rot_z"])
            tm.vertices = np.asarray(tm.vertices) @ Rz.T
            centers, rad = sphere_cover(tm, PITCH * so["scale"])
            c = np.array([DROP_XY[0], DROP_XY[1],
                          GZ + rad - float(centers[:, 2].min()) + LIFT])
            cache[obj] = (f"{cat}/{asset}", so["scale"], vmean, Rz, c)

            for cd, mu in itertools.product(CDS, MUS):
                key = f"{obj}_cd{int(cd)}_mu{int(mu*100):03d}"
                name, scale, vm, R0, c0 = cache[obj]
                s = ProbeScene([name], [list(c0)], [[0.0, 0.0, 0.0]],
                               densities=(DENSITY,), ground_z=GZ,
                               dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                               k=K_CONTACT, cd=float(cd), mu=float(mu),
                               mesh_scale=[scale], pitch=PITCH)
                s.rollout()
                P = s.positions(SUBSTEPS)[:NF]
                Q = s.rotations(SUBSTEPS)[:NF]
                if not np.isfinite(P).all():
                    rows.append({"key": key, "object": obj, "cd": cd, "mu": mu,
                                 "ok": False, "why": "non-finite"})
                    continue
                e, _v = restitution(P[:, 0, 2])
                a, b = event_window(P[:, 0, 2], fps=FPS, max_len=MAX_WIN)
                travel = float(np.linalg.norm(
                    P[a:b, 0, :2] - P[a, 0, :2], axis=1).max())
                ok = e is not None and (b - a) >= 12
                rows.append({"key": key, "object": obj, "cd": cd, "mu": mu,
                             "e": e, "travel_m": round(travel, 3),
                             "win": [a, b], "n_frames": b - a, "ok": bool(ok)})
                if not ok:
                    continue
                poses = []
                for t in range(a, b):
                    Rm = quat_to_mat(Q[t, 0]) @ R0
                    loc = P[t, 0] - Rm @ vm
                    poses.append({"loc": [float(x) for x in loc],
                                  "mat": [[float(v) for v in r] for r in Rm]})
                poses_out[key] = {"subject": obj, "poses": poses}

    (LAB / "sim_poses.json").write_text(json.dumps(poses_out))
    (LAB / "cells.json").write_text(json.dumps(rows, indent=2))
    shutil.copy(REPO / "outputs" / "judge" / "j2r" / "lab.json", LAB / "lab.json")
    good = [r for r in rows if r["ok"]]
    nf = [r["n_frames"] for r in good]
    print(f"{len(good)}/{len(rows)} usable in {time.time()-t0:.0f}s")
    print(f"  event window: {min(nf)}-{max(nf)} frames (median {int(np.median(nf))})")
    print(f"  total frames to render: {sum(nf)}")
    for obj in OBJS:
        e = [r["e"] for r in good if r["object"] == obj]
        t = [r["travel_m"] for r in good if r["object"] == obj]
        print(f"  {obj:<13} n={len(e):>2}  e {min(e):.3f}..{max(e):.3f}  "
              f"travel max {max(t):.2f} m")
    return 0


def do_encode():
    """Fold rendered frames into mp4s and run the motion guard on every one."""
    import imageio.v2 as imageio
    rows = {r["key"]: r for r in json.loads((LAB / "cells.json").read_text())}
    out = []
    for d in sorted(LAB.glob("sim_*")):
        if not d.is_dir():
            continue
        ps = sorted(d.glob("f*.png"))
        if not ps:
            continue
        key = d.name[4:]
        dst = LAB / f"{key}.mp4"
        w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                               macro_block_size=1)
        for p in ps:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        ok, frac, deltas = check(dst)
        r = dict(rows.get(key, {"key": key}))
        r.update(clip=dst.name, n_png=len(ps), motion_fraction=round(frac, 3),
                 guard_ok=bool(ok), deltas=[round(x, 2) for x in deltas])
        out.append(r)
    (LAB / "clips.json").write_text(json.dumps(out, indent=2))
    passed = [r for r in out if r["guard_ok"]]
    fr = [r["motion_fraction"] for r in out]
    print(f"{len(out)} clips encoded, guard passed {len(passed)}/{len(out)}")
    print(f"  motion fraction: min {min(fr):.2f}  median {np.median(fr):.2f}  "
          f"max {max(fr):.2f}")
    bad = [r for r in out if not r["guard_ok"]]
    for r in bad[:10]:
        print(f"  REJECT {r['clip']:<34} fraction {r['motion_fraction']:.2f}")
    print(f"\nwrote {LAB}/clips.json")
    return 0


MODES = {"sim": do_sim, "encode": do_encode}

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "sim"
    raise SystemExit(MODES[m]())
