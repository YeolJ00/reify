"""J3 helper: simulate a batch of theta candidates and write poses for Blender.

Runs in the WARP env. The J3 driver lives in the cosmos env (it holds the judge resident
in VRAM for the whole run), so simulation is shelled out one batch per CEM iteration
rather than one call per candidate -- Newton init costs more than the rollouts do.

Reads  <run>/batch_in.json   {key: {object, cd, mu}}
Writes <run>/sim_poses.json  {key: {subject, poses:[...]}}   (what Blender consumes)
       <run>/batch_out.json  {key: {e, travel_m, ok}}        (what the driver consumes)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    DENSITY, DROP_XY, FPS, K_CONTACT, LIFT, NF, PITCH, SUBSTEPS,
    quat_to_mat, restitution, rotz)

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

    # Mesh prep is per-object and independent of theta; cache it across the batch.
    cache = {}
    poses_out, meta_out = {}, {}
    with wp.ScopedDevice("cuda:0"):
        for key, th in todo.items():
            obj = th["object"]
            if obj not in cache:
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
            name, scale, vmean, Rz, c = cache[obj]

            s = ProbeScene([name], [list(c)], [[0.0, 0.0, 0.0]], densities=(DENSITY,),
                           ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                           n_steps=NF * SUBSTEPS, k=K_CONTACT, cd=float(th["cd"]),
                           mu=float(th["mu"]), mesh_scale=[scale], pitch=PITCH)
            s.rollout()
            P = s.positions(SUBSTEPS)[:NF]
            Q = s.rotations(SUBSTEPS)[:NF]
            e, _v = restitution(P[:, 0, 2])
            ok = e is not None and np.isfinite(P).all()
            travel = (float(np.linalg.norm(P[:, 0, :2] - P[0, 0, :2], axis=1).max())
                      if np.isfinite(P).all() else float("nan"))
            meta_out[key] = {"e": e, "travel_m": travel, "ok": bool(ok)}
            if not ok:
                continue
            poses = []
            for t in range(len(P)):
                R = quat_to_mat(Q[t, 0]) @ Rz
                loc = P[t, 0] - R @ vmean
                poses.append({"loc": [float(x) for x in loc],
                              "mat": [[float(v) for v in r] for r in R]})
            poses_out[key] = {"subject": obj, "poses": poses}

    (run / "sim_poses.json").write_text(json.dumps(poses_out))
    (run / "batch_out.json").write_text(json.dumps(meta_out))
    print(f"simulated {len(todo)}, usable {len(poses_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
