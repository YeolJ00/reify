"""Export a simulated rollout as per-frame object poses, for Blender to render.

The comparison used to be composited from sprites because I claimed there was no
photoreal renderer available. There is: Blender rendered every initial frame in this
project. Rendering the simulation in the same scene, lighting and camera removes every
compositing artefact by construction instead of patching them one at a time.

Placement is handled explicitly here, because the two conventions differ and getting that
wrong is what put the tracker off-object in the first place:

    the simulator's body sits at the mesh CENTROID
    Blender places the object's ORIGIN and then rotates it

so   blender_location = sim_centroid - R @ vmean   with R the full orientation.
The mesh is also rotated by the asset's rot_z before the sphere cover is built, so the
contact geometry matches the shape actually being rendered.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/export_sim_poses.py     (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"
SUBSTEPS, K_CONTACT, PITCH = 60, 2500.0, 0.020
NF, FPS = 49, 24.0
PICKS = [("ceramic_vase", "drop_mid", "restitution"),
         ("rubber_duck", "drop_mid", "restitution"),
         ("ceramic_vase", "slide", "friction")]


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    import warp as wp

    from src.data.assets import decimate, load_asset
    from src.motion.observables import drop_observables, slide_observables
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene
    from scripts.simple_fit import pixel_scale

    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds_image.json").read_text())
    fit = json.loads((LAB / "expand_fit.json").read_text())
    GZ = cfg["ground_z"]
    ppm, _ = pixel_scale(cfg["camera"], cfg["ground_z"])
    cam = Camera(cfg["camera"])
    wp.init()
    out = {}

    with wp.ScopedDevice("cuda:0"):
        for subj, stage, what in PICKS:
            key = f"{subj}_{stage}"
            if key not in cfg["experiments"]:
                continue
            e = cfg["experiments"][key]
            so = cfg["assets"][subj]
            cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
            tm = decimate(load_asset(cat, asset), 400).copy()
            tm.apply_scale(so["scale"])
            vmean_local = np.asarray(tm.vertices).mean(0).copy()
            # bake rot_z so the simulated shape is the shape being rendered
            Rz = rotz(so["rot_z"])
            tm.vertices = np.asarray(tm.vertices) @ Rz.T
            centers, rad = sphere_cover(tm, PITCH * so["scale"])
            c = np.array([e["subject_pos"][0], e["subject_pos"][1],
                          GZ + rad - float(centers[:, 2].min()) + e["lift"]])

            r = fit.get(subj, {})
            mu = (r.get("friction") or {}).get("value") or 0.4
            v0 = 0.0
            if what == "friction":
                ck = None
                for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
                    p = LAB / f"ptrk_img_{f.stem.replace('vid_','')}_subject.npz"
                    if p.exists():
                        ck = p; break
                if ck is not None:
                    d = np.load(ck)
                    o = slide_observables(np.stack([d["u"], d["v"]], 1), ppm, FPS)
                    v0 = abs(o.get("v0", 0.0)) * FPS / ppm if o.get("ok") else 0.0

            def run(cd):
                s = ProbeScene([f"{cat}/{asset}"], [list(c)], [[float(v0), 0.0, 0.0]],
                               densities=(600.0,), ground_z=GZ,
                               dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                               k=K_CONTACT, cd=float(cd), mu=float(mu),
                               mesh_scale=[so["scale"]], pitch=PITCH)
                s.rollout()
                return s.positions(SUBSTEPS)[:NF], s.rotations(SUBSTEPS)[:NF]

            cd = 20.0
            if what == "restitution" and r.get("e_mid") is not None:
                # bounded to the stable, monotonic region of the contact model: past
                # ~2m/dt the explicit penalty contact creates energy (measured e up to 9.9)
                lo, hi, best = 0.5, 200.0, None
                for _ in range(12):
                    t = float(np.sqrt(lo * hi))
                    P, Q = run(t)
                    uv, _ = cam.project(P[:, 0])
                    o = drop_observables(np.stack([uv[:, 0], uv[:, 1]], 1))
                    if not o.get("ok"):
                        hi = t; continue
                    if best is None or abs(o["value"] - r["e_mid"]) < abs(best[1] - r["e_mid"]):
                        best = (t, o["value"])
                    if o["value"] > r["e_mid"]:
                        lo = t
                    else:
                        hi = t
                cd = best[0] if best else 20.0
                note = (f"e measured {r['e_mid']:.3f}, simulated {best[1]:.3f}"
                        if best else "rollout failed")
            else:
                note = f"mu measured {mu:.3f}, launched at {v0:.2f} m/s"
            P, Q = run(cd)

            poses = []
            for t in range(len(P)):
                R = quat_to_mat(Q[t, 0]) @ Rz
                loc = P[t, 0] - R @ vmean_local
                poses.append({"loc": [float(x) for x in loc],
                              "mat": [[float(v) for v in row] for row in R]})
            out[key] = {"subject": subj, "note": note, "cd": float(cd),
                        "poses": poses}
            print(f"  {key}: {note}  ({len(poses)} frames)")

    (LAB / "sim_poses.json").write_text(json.dumps(out))
    print(f"\nwrote {LAB}/sim_poses.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
