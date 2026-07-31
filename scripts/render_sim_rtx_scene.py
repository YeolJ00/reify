"""Render the rollout in the STAGED SCENE with Newton's ViewerRTX. Warp-native, textured.

The earlier ViewerRTX pass drew a bare mesh on a default ground plane, which made it look
like Newton could not render properly. It can: `log_mesh` takes UVs, a texture, roughness
and metallic, so the same table, objects and materials the clips were staged with can be
rebuilt inside the viewer. Nothing here goes through Blender.

Layers, restated: Warp integrates a sphere-cover proxy and produces per-frame transforms;
the renderer needs textured triangles and a camera. They meet only at the transform, so the
simulated body's vertices are transformed per frame and re-logged, while the table and the
parked objects are logged once per frame unchanged.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/render_sim_rtx_scene.py    (warp env)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"
SCENE = REPO / "outputs" / "scene"
GROUND_Z = 0.706
PARK_Y = 0.46


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_angles(eye, target):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    return (math.degrees(math.asin(np.clip(f[2], -1, 1))),
            math.degrees(math.atan2(f[1], f[0])))


def mesh_bits(m):
    """vertices, triangle indices, uvs, texture as a numpy image (or None)."""
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int32).ravel()
    uv = getattr(m.visual, "uv", None)
    uv = None if uv is None else np.asarray(uv, np.float32)
    tex = None
    try:
        t = m.visual.material.baseColorTexture
        if t is not None:
            tex = np.asarray(t.convert("RGB"), np.uint8)
    except Exception:
        pass
    return V, F, uv, tex


def main():
    import warp as wp
    import newton.viewer as V

    from src.data.assets import load_asset

    poses = json.loads((LAB / "sim_poses.json").read_text())
    cfg = json.loads((LAB / "lab.json").read_text())
    scene = json.loads((SCENE / "scene.json").read_text())
    cam = cfg["camera"]
    wp.init()

    # table, placed exactly as the staging placed it
    tbl = load_asset("scenes", "wooden_table_02")
    tV, tF, tUV, tTex = mesh_bits(tbl)
    tV = tV + np.array([0.0, 0.0, GROUND_Z - tV[:, 2].max()], np.float32)

    made = {}
    for key, info in poses.items():
        subj = info["subject"]
        statics = []
        for name, o in scene["objects"].items():
            m = load_asset(*o["asset"].split("/")[-2:])
            Vv, Ff, Uu, Tt = mesh_bits(m)
            Vv = (Vv * o["scale"]) @ rotz(o["rot_z"]).T
            pos = np.array(o["pos"], np.float32)
            if name != subj:
                pos = pos + np.array([0.0, PARK_Y, 0.0], np.float32)
                statics.append((name, Vv + pos, Ff, Uu, Tt))
            else:
                sub = (Vv, Ff, Uu, Tt)

        viewer = V.ViewerRTX(width=int(cam["width"]), height=int(cam["height"]),
                             headless=True, up_axis="Z", environment="studio")
        pitch, yaw = camera_angles(cam["eye"], cam["target"])
        viewer.set_camera(wp.vec3(*[float(x) for x in cam["eye"]]), pitch, yaw)

        outdir = LAB / f"rtx_{key}"
        outdir.mkdir(exist_ok=True)
        sV, sF, sUV, sTex = sub
        com = sV.mean(0)
        for t, p in enumerate(info["poses"]):
            R = np.asarray(p["mat"], np.float32)
            loc = np.asarray(p["loc"], np.float32)
            viewer.begin_frame(t / 24.0)
            viewer.log_mesh("table", wp.array(tV, dtype=wp.vec3),
                            wp.array(tF, dtype=wp.int32),
                            uvs=None if tUV is None else wp.array(tUV, dtype=wp.vec2),
                            texture=tTex)
            for nm, Vv, Ff, Uu, Tt in statics:
                viewer.log_mesh(nm, wp.array(Vv, dtype=wp.vec3),
                                wp.array(Ff, dtype=wp.int32),
                                uvs=None if Uu is None else wp.array(Uu, dtype=wp.vec2),
                                texture=Tt)
            # the simulated body: vertices carried to the rollout's pose this frame
            Vw = (sV - com) @ R.T + (loc + R @ com)
            viewer.log_mesh(subj, wp.array(Vw.astype(np.float32), dtype=wp.vec3),
                            wp.array(sF, dtype=wp.int32),
                            uvs=None if sUV is None else wp.array(sUV, dtype=wp.vec2),
                            texture=sTex)
            viewer.end_frame()
            viewer.save_screenshot(str(outdir / f"f{t:04d}.png"))
        viewer.close()
        made[key] = {"note": info["note"], "n": len(info["poses"]), "dir": outdir.name}
        print(f"  {key}: {len(info['poses'])} frames -> {outdir.name}")

    (LAB / "rtx_render.json").write_text(json.dumps(made, indent=2))
    print(f"\nrendered {len(made)} rollouts in the staged scene with ViewerRTX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
