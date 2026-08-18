"""Export the balance rig as loadable assets + poses, so the normal render path works.

The rig is procedural (beam, pans, weights are trimesh primitives) but
blender_render_scene.py imports assets from disk by name. Writing the primitives out as OBJs
into assets/rigid/ is enough -- import_any already handles OBJ, and the pose file format is
the same one the tilt scene uses.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from probe_balance import (ARM, BEAM_T, GZ, PIVOT_H, beam_with_pans,  # noqa: E402
                           run, settled, stand_mesh)

LAB = Path(os.environ.get("LAB", "outputs/scene/balance"))
CASES = [("light_ref", 0.30, 0.10), ("balanced", 0.30, 0.30), ("heavy_ref", 0.30, 0.60)]


def export(mesh, name):
    d = REPO / "assets" / "rigid" / name / "meshes"
    d.mkdir(parents=True, exist_ok=True)
    mesh.export(d / "model.obj")
    return f"rigid/{name}"


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    W = 0.05
    a_beam = export(beam_with_pans(), "_bal_beam")
    a_stand = export(stand_mesh(), "_bal_stand")
    a_wt = export(trimesh.creation.box(extents=(W,) * 3), "_bal_weight")
    # Must match probe_balance.run()'s body order exactly: beam, stand, unknown, reference.
    # A hard-coded three-name list shifted every body one slot, so the stand rendered as a
    # weight box and the reference mass vanished from the scene entirely.
    BODIES = [("beam", a_beam), ("stand", a_stand), ("obj", a_wt), ("ref", a_wt)]
    scenes, objs = {}, {}
    with wp.ScopedDevice("cuda:0"):
        for tag, mo, mr in CASES:
            ang, s = run(mo, mr)
            P, Q = s.positions(60), s.rotations(60)
            key = f"balance_{tag}"
            scenes[key] = {"tilt_deg": 0.0, "objects": {}}
            assert len(s.coms) == len(BODIES), f"{len(s.coms)} bodies vs {len(BODIES)} names"
            for i, (nm, asset) in enumerate(BODIES):
                # COM-centred body frame -> mesh origin, same convention as bigscene_sim
                C = s.coms[i]
                seq = []
                for t in range(len(P)):
                    R = quat_to_mat(Q[t, i])
                    seq.append({"loc": [float(v) for v in (P[t, i] - R @ C)],
                                "mat": [[float(v) for v in r] for r in R]})
                nk = f"{tag}_{nm}"
                scenes[key]["objects"][nk] = seq
                objs[nk] = {"asset": asset, "scale": 1.0, "rot_z": 0.0,
                            "pos": [float(v) for v in P[0, i]]}
            print(f"  {tag:<10} m_obj={mo} m_ref={mr}  settled {settled(ang):+6.2f} deg")
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(json.dumps({"ground_z": GZ, "objects": objs}, indent=1))
    print(f"  wrote {LAB}/scene_poses.json  ({len(scenes)} scenes, {len(objs)} bodies)")


if __name__ == "__main__":
    main()
