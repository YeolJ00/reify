"""Simulate all three probe scenes for a set of theta. Runs in the WARP env.

  joint_sim.py <run>         reads <run>/joint_in.json -> scene_poses.json
  joint_sim.py <run> crop    rendered frames -> per-object clips + crops.json

Probes, and what each one constrains:
  tilt     a ramp from 4 to 34 deg. Slip ANGLE reads friction; topple angle reads centre of
           mass. Both are thresholds and both are mass-independent.
  drop     0.22 m onto the table. Bounce reads cd/m -- the RATIO only, which is why a drop
           alone cannot separate damping from mass.
  collide  the object is launched at a stationary partner of known mass. Momentum
           transfer sees MASS directly, which is what breaks the drop's degeneracy.
           Each subject gets its OWN partner, since a Blender scene holds one object per name.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import (  # noqa: E402
    FPS, K_CONTACT, PITCH, SUBSTEPS, quat_to_mat, rotz)
from scripts.multiprobe_build import GZ, PIVOT, prep, ry  # noqa: E402
from src.sim.tilt_probe import tilt_gravity  # noqa: E402

SRC = REPO / "outputs" / "scene" / "expand"
# Six experiments. Weight comes from measured informativeness, not from intuition -- the
# tilt probe is the most sign-consistent and the LEAST informative, because it is saturated
# (mean p(yes) 0.948, so p(1-p) = 0.047). Those are different properties.
PROBES = ("tilt", "drop", "collide", "spin", "stack", "shove")
TILT_A, TILT_B, TILT_N = 4.0, 34.0, 44
DROP_LIFT, DROP_N = 0.22, 34
COLL_V, COLL_N = 1.5, 30          # launch speed m/s, frames
SPIN_W, SPIN_N = 9.0, 34          # initial spin rad/s, frames
STACK_N = 44                      # stack on the ramp, same schedule as tilt
SHOVE_V, SHOVE_N = 1.1, 30        # flat-ground shove, frames
# One partner PER SUBJECT. Both subjects share a collide scene, and a Blender scene holds
# one object of each name -- keying the partner by a single name meant the second collision
# silently overwrote the first and only one partner appeared in the render.
PARTNERS = {"brass_pot": "baseball", "wooden_bowl": "apple"}


def _poses(P, Q, Rz, vm, ramp=None):
    out = []
    for t in range(len(P)):
        R = quat_to_mat(Q[t]) @ Rz
        loc = P[t] - R @ vm
        if ramp is not None:
            Rt = ry(np.deg2rad(ramp[min(t, len(ramp) - 1)]))
            loc = PIVOT + Rt @ (loc - PIVOT)
            R = Rt @ R
        out.append({"loc": [float(x) for x in loc],
                    "mat": [[float(v) for v in r] for r in R]})
    return out


def do_sim(run):
    import warp as wp
    from src.sim.probe_scene import ProbeScene
    spec = json.loads((run / "joint_in.json").read_text())
    cfg = json.loads((SRC / "lab.json").read_text())
    wp.init()
    ramp = np.linspace(TILT_A, TILT_B, TILT_N)
    scenes, meta, cache = {}, {}, {}

    with wp.ScopedDevice("cuda:0"):
        def P(o):
            if o not in cache:
                cache[o] = prep(cfg, o)
            return cache[o]

        for tag, objs in spec.items():          # tag = "plus" / "minus"
            for probe in PROBES:
                key = f"{tag}_{probe}"
                sob, smet = {}, {}
                for o, th in objs.items():
                    name, sc, vm, Rz, rad, zmin = P(o)
                    rho, mu, cd = th["rho"], th["mu"], th["cd"]
                    y = th["y"]
                    if probe == "tilt":
                        pos = [PIVOT[0] - 0.10, y, GZ + rad - zmin + 0.002]
                        vel = [0.0, 0.0, 0.0]
                        Ps, Qs = [], []
                        for f in range(TILT_N):
                            s = ProbeScene([name], [pos], [vel], densities=(rho,),
                                           ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                                           n_steps=SUBSTEPS, k=K_CONTACT, cd=cd, mu=mu,
                                           mesh_scale=[sc], pitch=PITCH,
                                           gravity=tilt_gravity(np.deg2rad(ramp[f])))
                            s.rollout()
                            A, B = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                            if not np.isfinite(A).all():
                                break
                            Ps.append(A[-1, 0].copy()); Qs.append(B[-1, 0].copy())
                            vel = [float(v) for v in (A[-1, 0] - A[0, 0]) * FPS]
                            pos = [float(v) for v in A[-1, 0]]
                        if len(Ps) < TILT_N:
                            continue
                        Pa, Qa = np.array(Ps), np.array(Qs)
                        sob[o] = _poses(Pa, Qa, Rz, vm, ramp=ramp)
                        smet[o] = {"world_pts": Pa.tolist(), "ramp": ramp.tolist()}
                    elif probe == "drop":
                        c = [PIVOT[0], y, GZ + rad - zmin + DROP_LIFT]
                        s = ProbeScene([name], [c], [[0.0, 0.0, 0.0]], densities=(rho,),
                                       ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                                       n_steps=DROP_N * SUBSTEPS, k=K_CONTACT, cd=cd,
                                       mu=mu, mesh_scale=[sc], pitch=PITCH)
                        s.rollout()
                        A = s.positions(SUBSTEPS)[:DROP_N]
                        B = s.rotations(SUBSTEPS)[:DROP_N]
                        if not np.isfinite(A).all():
                            continue
                        sob[o] = _poses(A[:, 0], B[:, 0], Rz, vm)
                        smet[o] = {"world_pts": A[:, 0].tolist()}
                    elif probe == "spin":
                        # spun on the spot. How long it keeps turning reads friction, and
                        # the decay is a RATE rather than a distance, so it is not a
                        # straight motion-magnitude cue.
                        c = [PIVOT[0], y, GZ + rad - zmin + 0.002]
                        s = ProbeScene([name], [c], [[0.0, 0.0, 0.0]],
                                       ang0=[[0.0, 0.0, SPIN_W]], densities=(rho,),
                                       ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                                       n_steps=SPIN_N * SUBSTEPS, k=K_CONTACT, cd=cd,
                                       mu=mu, mesh_scale=[sc], pitch=PITCH)
                        s.rollout()
                        A = s.positions(SUBSTEPS)[:SPIN_N]
                        B = s.rotations(SUBSTEPS)[:SPIN_N]
                        if not np.isfinite(A).all():
                            continue
                        sob[o] = _poses(A[:, 0], B[:, 0], Rz, vm)
                        smet[o] = {"world_pts": A[:, 0].tolist()}
                    elif probe == "stack":
                        # a partner balanced on top, then the ramp. The stack falls at ONE
                        # angle -- a binary outcome, which is where a yes/no judge carries
                        # the most information, and it reads CoM and friction jointly.
                        partner = PARTNERS.get(o, "baseball")
                        pn, psc, pvm, pRz, prad, pzmin = P(partner)
                        base_z = GZ + rad - zmin + 0.002
                        top_z = base_z + (zmin + rad) * 0.0 + 0.16
                        pos = [[PIVOT[0] - 0.05, y, base_z], [PIVOT[0] - 0.05, y, top_z]]
                        vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        Ps, Qs, P2, Q2 = [], [], [], []
                        for f in range(STACK_N):
                            s = ProbeScene([name, pn], pos, vel, densities=(rho, 650.0),
                                           ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                                           n_steps=SUBSTEPS, k=K_CONTACT, cd=cd, mu=mu,
                                           mesh_scale=[sc, psc], pitch=PITCH,
                                           gravity=tilt_gravity(np.deg2rad(ramp[f])))
                            s.rollout()
                            A, B = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                            if not np.isfinite(A).all():
                                break
                            Ps.append(A[-1, 0].copy()); Qs.append(B[-1, 0].copy())
                            P2.append(A[-1, 1].copy()); Q2.append(B[-1, 1].copy())
                            vel = [[float(v) for v in (A[-1, i] - A[0, i]) * FPS]
                                   for i in (0, 1)]
                            pos = [[float(v) for v in A[-1, i]] for i in (0, 1)]
                        if len(Ps) < STACK_N:
                            continue
                        sob[o] = _poses(np.array(Ps), np.array(Qs), Rz, vm, ramp=ramp)
                        sob[partner] = _poses(np.array(P2), np.array(Q2), pRz, pvm,
                                              ramp=ramp)
                        smet[o] = {"world_pts": np.array(Ps).tolist(), "partner": partner}
                    elif probe == "shove":
                        # pushed across a level surface and left to stop. Stopping is a
                        # threshold event even though the distance is a magnitude.
                        c = [PIVOT[0] - 0.22, y, GZ + rad - zmin + 0.002]
                        s = ProbeScene([name], [c], [[SHOVE_V, 0.0, 0.0]],
                                       densities=(rho,), ground_z=GZ,
                                       dt=1.0 / (FPS * SUBSTEPS),
                                       n_steps=SHOVE_N * SUBSTEPS, k=K_CONTACT, cd=cd,
                                       mu=mu, mesh_scale=[sc], pitch=PITCH)
                        s.rollout()
                        A = s.positions(SUBSTEPS)[:SHOVE_N]
                        B = s.rotations(SUBSTEPS)[:SHOVE_N]
                        if not np.isfinite(A).all():
                            continue
                        sob[o] = _poses(A[:, 0], B[:, 0], Rz, vm)
                        smet[o] = {"world_pts": A[:, 0].tolist()}
                    else:                        # collide
                        partner = PARTNERS.get(o, "baseball")
                        pn, psc, pvm, pRz, prad, pzmin = P(partner)
                        c0 = [PIVOT[0] - 0.28, y, GZ + rad - zmin + 0.002]
                        c1 = [PIVOT[0] + 0.10, y, GZ + prad - pzmin + 0.002]
                        s = ProbeScene([name, pn], [c0, c1],
                                       [[COLL_V, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                       densities=(rho, 650.0), ground_z=GZ,
                                       dt=1.0 / (FPS * SUBSTEPS),
                                       n_steps=COLL_N * SUBSTEPS, k=K_CONTACT, cd=cd,
                                       mu=mu, mesh_scale=[sc, psc], pitch=PITCH)
                        s.rollout()
                        A = s.positions(SUBSTEPS)[:COLL_N]
                        B = s.rotations(SUBSTEPS)[:COLL_N]
                        if not np.isfinite(A).all():
                            continue
                        sob[o] = _poses(A[:, 0], B[:, 0], Rz, vm)
                        sob[partner] = _poses(A[:, 1], B[:, 1], pRz, pvm)
                        smet[o] = {"world_pts": A[:, 0].tolist(),
                                   "partner": partner,
                                   "partner_pts": A[:, 1].tolist()}
                if sob:
                    scenes[key] = {"tilt_deg": ramp.tolist()
                                   if probe in ("tilt", "stack") else 0.0,
                                   "objects": sob}
                    meta[key] = smet
    (run / "scene_poses.json").write_text(json.dumps(scenes))
    (run / "scene_meta.json").write_text(json.dumps(meta))
    print(f"simulated {len(scenes)} scenes")
    return 0


def do_crop(run):
    import imageio.v2 as imageio
    from src.render.camera import Camera
    from src.render.crop import crop_box, crop_clip
    meta = json.loads((run / "scene_meta.json").read_text())
    scenes = json.loads((run / "scene_poses.json").read_text())
    out = []
    views = sorted({d.name.split("@")[1] for d in run.glob("sim_*@*") if d.is_dir()})
    cams = {}
    lab = json.loads((run / "lab.json").read_text())
    from src.render.views import VIEWS as VDEF   # shared, bpy-free
    for v in views:
        cams[v] = Camera(VDEF[v])
    for key, sc in scenes.items():
      for v in views:
        ps = sorted((run / f"sim_{key}@{v}").glob("f*.png"))
        if not ps:
            continue
        cam = cams[v]
        frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
        td = sc["tilt_deg"]
        for o, m in meta.get(key, {}).items():
            pts = np.asarray(m["world_pts"], float)
            if isinstance(td, list):
                pts = np.array([PIVOT + ry(np.deg2rad(td[min(i, len(td) - 1)])) @ (p - PIVOT)
                                for i, p in enumerate(pts)])
            pts = pts[:, None, :]
            box = crop_box(cam, pts, 544, 448)
            if box is None:
                continue
            dst = run / f"{key}__{o}@{v}.mp4"
            w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                                   macro_block_size=1)
            for f in crop_clip(frames, box):
                w.append_data(f)
            w.close()
            out.append({"scene": key, "tag": key.split("_")[0], "view": v,
                        "probe": key.split("_", 1)[1], "object": o, "clip": dst.name})
    (run / "crops.json").write_text(json.dumps(out, indent=2))
    print(f"cropped {len(out)} clips")
    return 0


if __name__ == "__main__":
    run = Path(sys.argv[1])
    raise SystemExit(do_crop(run) if len(sys.argv) > 2 and sys.argv[2] == "crop"
                     else do_sim(run))
