"""Sim + crop helper for the tilt SPSA fit. Runs in the WARP env.

  spsa_tilt_sim.py <run>          reads <run>/spsa_in.json -> writes scene_poses.json
  spsa_tilt_sim.py <run> crop     reads the rendered frames -> writes crops.json + clips
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    DENSITY, FPS, K_CONTACT, PITCH, SUBSTEPS, quat_to_mat, rotz)
from scripts.multiprobe_build import (  # noqa: E402
    GZ, PIVOT, TILT_DEG_END, TILT_DEG_START, TILT_FRAMES, prep, ry)
from src.sim.tilt_probe import tilt_gravity  # noqa: E402

SRC = REPO / "outputs" / "scene" / "expand"


def do_sim(run):
    import warp as wp
    from src.sim.probe_scene import ProbeScene

    spec = json.loads((run / "spsa_in.json").read_text())
    cfg = json.loads((SRC / "lab.json").read_text())
    wp.init()
    ramp = np.linspace(TILT_DEG_START, TILT_DEG_END, TILT_FRAMES)
    scenes, meta = {}, {}
    with wp.ScopedDevice("cuda:0"):
        cache = {}
        for scene, objs in spec.items():
            sobjs, smeta = {}, {}
            for obj, d in objs.items():
                if obj not in cache:
                    cache[obj] = prep(cfg, obj)
                name, sc, vm, Rz, rad, zmin = cache[obj]
                pos = [PIVOT[0] - 0.10, d["y"], GZ + rad - zmin + 0.002]
                vel = [0.0, 0.0, 0.0]
                Ps, Qs = [], []
                for f in range(TILT_FRAMES):
                    s = ProbeScene([name], [pos], [vel], densities=(DENSITY,),
                                   ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                                   n_steps=SUBSTEPS, k=K_CONTACT, cd=3000.0,
                                   mu=float(d["mu"]), mesh_scale=[sc], pitch=PITCH,
                                   gravity=tilt_gravity(np.deg2rad(ramp[f])))
                    s.rollout()
                    P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                    if not np.isfinite(P).all():
                        break
                    Ps.append(P[-1, 0].copy()); Qs.append(Q[-1, 0].copy())
                    vel = [float(v) for v in (P[-1, 0] - P[0, 0]) * FPS]
                    pos = [float(v) for v in P[-1, 0]]
                if len(Ps) < TILT_FRAMES:
                    continue
                P = np.array(Ps); Q = np.array(Qs)
                dist = np.linalg.norm(P[:, :2] - P[0, :2], axis=1)
                on = next((f for f in range(TILT_FRAMES) if dist[f] > 0.012), None)
                seq = []
                for t in range(TILT_FRAMES):
                    R = quat_to_mat(Q[t]) @ Rz
                    loc = P[t] - R @ vm
                    Rt = ry(np.deg2rad(ramp[t]))
                    seq.append({"loc": [float(z) for z in (PIVOT + Rt @ (loc - PIVOT))],
                                "mat": [[float(v) for v in r] for r in (Rt @ R)]})
                sobjs[obj] = seq
                smeta[obj] = {"mu": d["mu"], "onset_deg":
                              float(ramp[on]) if on is not None else None,
                              "world_pts": P.tolist()}
            scenes[scene] = {"tilt_deg": ramp.tolist(), "objects": sobjs}
            meta[scene] = smeta
    (run / "scene_poses.json").write_text(json.dumps(scenes))
    (run / "scene_meta.json").write_text(json.dumps(meta))
    print(f"simulated {sum(len(v['objects']) for v in scenes.values())} rollouts")
    return 0


def do_crop(run):
    import imageio.v2 as imageio
    from src.render.camera import Camera
    from src.render.crop import crop_box, crop_clip, occupancy

    cam = Camera(json.loads((run / "lab.json").read_text())["camera"])
    meta = json.loads((run / "scene_meta.json").read_text())
    scenes = json.loads((run / "scene_poses.json").read_text())
    out = []
    for key, sc in scenes.items():
        ps = sorted((run / f"sim_{key}").glob("f*.png"))
        if not ps:
            continue
        frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
        td = sc["tilt_deg"]
        for obj in sc["objects"]:
            pts = np.asarray(meta[key][obj]["world_pts"], float)
            pts = np.array([PIVOT + ry(np.deg2rad(td[min(i, len(td) - 1)])) @ (p - PIVOT)
                            for i, p in enumerate(pts)])[:, None, :]
            box = crop_box(cam, pts, 544, 448)
            if box is None:
                continue
            dst = run / f"{key}__{obj}.mp4"
            w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                                   macro_block_size=1)
            for f in crop_clip(frames, box):
                w.append_data(f)
            w.close()
            out.append({"scene": key, "object": obj, "clip": dst.name, "box": box,
                        "mu": meta[key][obj]["mu"],
                        "onset_deg": meta[key][obj]["onset_deg"],
                        "motion_in_crop": round(occupancy(frames, box), 3)})
    (run / "crops.json").write_text(json.dumps(out, indent=2))
    print(f"cropped {len(out)} clips")
    return 0


if __name__ == "__main__":
    run = Path(sys.argv[1])
    raise SystemExit(do_crop(run) if len(sys.argv) > 2 and sys.argv[2] == "crop"
                     else do_sim(run))
