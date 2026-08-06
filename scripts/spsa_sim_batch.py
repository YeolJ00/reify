"""Simulate a batch of theta candidates for the SPSA fit and write poses for Blender.

Matches the band sweep exactly -- 0.30 m drop, trimmed to the event window -- so the clips
the optimiser is scored on are drawn from the same distribution the calibration used.

Reads  <run>/batch_in.json   {key: {object, cd, mu}}
Writes <run>/sim_poses.json  and  <run>/batch_out.json  {key: {e, bounces, travel_m, ok}}
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    DENSITY, DROP_XY, FPS, K_CONTACT, PITCH, SUBSTEPS, quat_to_mat, restitution, rotz)
from scripts.g0c_build_clips import LIFT, NF, MAX_WIN  # noqa: E402
from scripts.band_build import bounces  # noqa: E402
from src.render.motion_budget import event_window  # noqa: E402

SRC = REPO / "outputs" / "scene" / "expand"


def main():
    run = Path(sys.argv[1])
    todo = json.loads((run / "batch_in.json").read_text())

    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    cfg = json.loads((SRC / "lab.json").read_text())
    GZ = cfg["ground_z"]
    wp.init()
    cache, poses, meta = {}, {}, {}
    with wp.ScopedDevice("cuda:0"):
        for key, th in todo.items():
            obj = th["object"]
            if obj not in cache:
                so = cfg["assets"][obj]
                cat, a = so["asset"].split("/")[0], so["asset"].split("/")[-1]
                tm = decimate(load_asset(cat, a), 400).copy()
                tm.apply_scale(so["scale"])
                vm = np.asarray(tm.vertices).mean(0).copy()
                Rz = rotz(so["rot_z"])
                tm.vertices = np.asarray(tm.vertices) @ Rz.T
                ctr, rad = sphere_cover(tm, PITCH * so["scale"])
                c = np.array([DROP_XY[0], DROP_XY[1],
                              GZ + rad - float(ctr[:, 2].min()) + LIFT])
                cache[obj] = (f"{cat}/{a}", so["scale"], vm, Rz, c)
            name, scale, vm, Rz, c = cache[obj]
            s = ProbeScene([name], [list(c)], [[0.0, 0.0, 0.0]], densities=(DENSITY,),
                           ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                           n_steps=NF * SUBSTEPS, k=K_CONTACT, cd=float(th["cd"]),
                           mu=float(th["mu"]), mesh_scale=[scale], pitch=PITCH)
            s.rollout()
            P = s.positions(SUBSTEPS)[:NF]
            Q = s.rotations(SUBSTEPS)[:NF]
            if not np.isfinite(P).all():
                meta[key] = {"ok": False}
                continue
            e, _ = restitution(P[:, 0, 2])
            lo, hi = event_window(P[:, 0, 2], fps=FPS, max_len=MAX_WIN)
            ok = e is not None and (hi - lo) >= 12
            meta[key] = {"e": e, "bounces": bounces(P[lo:hi, 0, 2]),
                         "travel_m": round(float(np.linalg.norm(
                             P[lo:hi, 0, :2] - P[lo, 0, :2], axis=1).max()), 3),
                         "ok": bool(ok)}
            if ok:
                poses[key] = {"subject": obj, "poses": [
                    {"loc": [float(x) for x in (P[t, 0] - (quat_to_mat(Q[t, 0]) @ Rz) @ vm)],
                     "mat": [[float(v) for v in r] for r in (quat_to_mat(Q[t, 0]) @ Rz)]}
                    for t in range(lo, hi)]}
    (run / "sim_poses.json").write_text(json.dumps(poses))
    (run / "batch_out.json").write_text(json.dumps(meta))
    print(f"simulated {len(todo)}, usable {len(poses)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
