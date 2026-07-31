"""Render the simulated rollout with Newton's own viewer (ViewerRTX, headless).

Replaces the hand-written rasteriser. ProbeScene is pure Warp -- custom 6-DOF kernels,
never a newton.Model -- which is why Newton's viewers had nothing to draw. That is a
reason to give Newton a Model, not a reason to write a third renderer: the geometry is
loaded into a newton.ModelBuilder here purely for display, and the body transforms are
driven frame by frame from our own rollout.

Requires `ovrtx` and `pyglet` (both installed into the warp env for this).

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/render_sim_newton.py    (warp env)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"


def camera_angles(eye, target):
    """Newton's set_camera takes pitch/yaw; our lab camera is an eye/target pair."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    yaw = math.degrees(math.atan2(f[1], f[0]))
    pitch = math.degrees(math.asin(np.clip(f[2], -1.0, 1.0)))
    return pitch, yaw


def main():
    import warp as wp
    import newton
    import newton.viewer as V

    from src.data.assets import decimate, load_asset

    poses = json.loads((LAB / "sim_poses.json").read_text())
    cfg = json.loads((LAB / "lab.json").read_text())
    cam = cfg["camera"]; GZ = cfg["ground_z"]
    wp.init()
    made = {}

    for key, info in poses.items():
        subj = info["subject"]
        so = cfg["assets"][subj]
        cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
        tm = decimate(load_asset(cat, asset), 1500).copy()
        tm.apply_scale(so["scale"])
        verts = np.asarray(tm.vertices, np.float32)
        com = verts.mean(0)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, GZ),
                                                   wp.quat_identity()))
        mesh = newton.Mesh((verts - com).astype(np.float32),
                           np.asarray(tm.faces, np.int32).ravel())
        builder.add_shape_mesh(body, mesh=mesh,
                               cfg=newton.ModelBuilder.ShapeConfig(density=600.0))
        model = builder.finalize()
        state = model.state()

        viewer = V.ViewerRTX(width=int(cam["width"]), height=int(cam["height"]),
                             headless=True, up_axis="Z", environment="studio")
        viewer.set_model(model)
        pitch, yaw = camera_angles(cam["eye"], cam["target"])
        viewer.set_camera(wp.vec3(*[float(x) for x in cam["eye"]]), pitch, yaw)

        outdir = LAB / f"nsim_{key}"
        outdir.mkdir(exist_ok=True)
        bq = state.body_q.numpy()
        for t, p in enumerate(info["poses"]):
            R = np.asarray(p["mat"], float)
            # sim_poses stores the Blender origin; Newton's body is centred on the COM
            pos = np.asarray(p["loc"], float) + R @ com
            q = wp.quat_from_matrix(wp.mat33(*[float(v) for v in R.flatten()]))
            bq[0] = [pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3]]
            state.body_q.assign(bq)
            viewer.begin_frame(t / 24.0)
            viewer.log_state(state)
            viewer.end_frame()
            viewer.save_screenshot(str(outdir / f"f{t:04d}.png"))
        viewer.close()
        made[key] = {"note": info["note"], "n": len(info["poses"]),
                     "dir": outdir.name}
        print(f"  {key}: {len(info['poses'])} frames -> {outdir.name}")

    (LAB / "newton_render.json").write_text(json.dumps(made, indent=2))
    print(f"\nrendered {len(made)} rollouts with Newton ViewerRTX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
