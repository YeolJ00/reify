"""Re-centred sweep: restitution in the PLAUSIBLE band, not the superball band.

The previous grid ran e up to 0.81, where ground-truth bounce counts are 17-21. A real
rubber duck does not bounce 18 times, so "less bouncy is more plausible" was arguably the
correct judgement across that whole range for BOTH objects -- which is why no sign flip
could appear and why the monotone statistic found only motion magnitude.

This grid spans e ~ 0.03..0.30, where the same objects bounce roughly 1-8 times. Damping
values were chosen by measuring e(cd) first; above cd ~5800 the mapping folds back on
itself, so the grid stays in the monotone region and the axis is the MEASURED e regardless.

Run:
  CUDA_VISIBLE_DEVICES=<g> .../envs/warp/bin/python scripts/band_build.py sim
  LAB=outputs/judge/band <blender> --background --python scripts/blender_render_sim.py
  .../envs/warp/bin/python scripts/band_build.py encode
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
from scripts.g0c_build_clips import LIFT, NF, MAX_WIN  # noqa: E402
from src.render.motion_budget import event_window, check  # noqa: E402

LAB = REPO / "outputs" / "judge" / "band"
SRC = REPO / "outputs" / "scene" / "expand"
OBJS = ("rubber_duck", "brass_pot")
CDS = [2000.0, 2600.0, 3400.0, 4400.0, 5800.0, 7500.0, 10000.0]
MUS = [0.20, 0.40, 0.70, 1.00]


def bounces(z, min_rise=0.004):
    z = np.asarray(z, float)
    n, i = 0, 1
    while i < len(z) - 1:
        if z[i] <= z[i - 1] and z[i] < z[i + 1]:
            j = i + 1
            while j < len(z) - 1 and z[j + 1] >= z[j]:
                j += 1
            if z[j] - z[i] >= min_rise:
                n += 1
            i = j
        i += 1
    return n


def do_sim():
    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    LAB.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "lab.json").read_text())
    GZ = cfg["ground_z"]
    wp.init()
    poses_out, rows, t0 = {}, [], time.time()
    with wp.ScopedDevice("cuda:0"):
        for obj in OBJS:
            so = cfg["assets"][obj]
            cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
            tm = decimate(load_asset(cat, asset), 400).copy()
            tm.apply_scale(so["scale"])
            vm = np.asarray(tm.vertices).mean(0).copy()
            Rz = rotz(so["rot_z"])
            tm.vertices = np.asarray(tm.vertices) @ Rz.T
            ctr, rad = sphere_cover(tm, PITCH * so["scale"])
            c = np.array([DROP_XY[0], DROP_XY[1],
                          GZ + rad - float(ctr[:, 2].min()) + LIFT])
            for cd, mu in itertools.product(CDS, MUS):
                key = f"{obj}_cd{int(cd)}_mu{int(mu*100):03d}"
                s = ProbeScene([f"{cat}/{asset}"], [list(c)], [[0.0, 0.0, 0.0]],
                               densities=(DENSITY,), ground_z=GZ,
                               dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                               k=K_CONTACT, cd=float(cd), mu=float(mu),
                               mesh_scale=[so["scale"]], pitch=PITCH)
                s.rollout()
                P = s.positions(SUBSTEPS)[:NF]
                Q = s.rotations(SUBSTEPS)[:NF]
                if not np.isfinite(P).all():
                    continue
                e, _ = restitution(P[:, 0, 2])
                a, b = event_window(P[:, 0, 2], fps=FPS, max_len=MAX_WIN)
                ok = e is not None and (b - a) >= 12
                rows.append({"key": key, "object": obj, "cd": cd, "mu": mu,
                             "e": e, "bounces": bounces(P[a:b, 0, 2]),
                             "travel_m": round(float(np.linalg.norm(
                                 P[a:b, 0, :2] - P[a, 0, :2], axis=1).max()), 3),
                             "n_frames": b - a, "ok": bool(ok)})
                if not ok:
                    continue
                poses_out[key] = {"subject": obj, "poses": [
                    {"loc": [float(x) for x in (P[t, 0] - (quat_to_mat(Q[t, 0]) @ Rz) @ vm)],
                     "mat": [[float(v) for v in r]
                             for r in (quat_to_mat(Q[t, 0]) @ Rz)]}
                    for t in range(a, b)]}
    (LAB / "sim_poses.json").write_text(json.dumps(poses_out))
    (LAB / "cells.json").write_text(json.dumps(rows, indent=2))
    shutil.copy(REPO / "outputs" / "judge" / "g0c" / "lab.json", LAB / "lab.json")
    good = [r for r in rows if r["ok"]]
    print(f"{len(good)}/{len(rows)} usable in {time.time()-t0:.0f}s, "
          f"{sum(r['n_frames'] for r in good)} frames to render")
    for obj in OBJS:
        g = [r for r in good if r["object"] == obj]
        e = [r["e"] for r in g]
        b = [r["bounces"] for r in g]
        print(f"  {obj:<13} n={len(g):>2}  e {min(e):.3f}..{max(e):.3f}  "
              f"bounces {min(b)}..{max(b)}  (old grid was e..0.81, bounces..21)")
    return 0


def do_encode():
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
        ok, frac, _d = check(dst)
        r = dict(rows.get(key, {"key": key}))
        r.update(clip=dst.name, motion_fraction=round(frac, 3), guard_ok=bool(ok))
        out.append(r)
    (LAB / "clips.json").write_text(json.dumps(out, indent=2))
    print(f"{len(out)} clips, guard passed {sum(1 for r in out if r['guard_ok'])}")
    return 0


MODES = {"sim": do_sim, "encode": do_encode}
if __name__ == "__main__":
    raise SystemExit(MODES[sys.argv[1] if len(sys.argv) > 1 else "sim"]())
