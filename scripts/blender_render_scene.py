"""Render a MULTI-OBJECT probe scene: several objects animated in one clip.

blender_render_sim.py animates one subject and parks the rest, which costs one render per
object. This renders every participating object in a single clip so the frames can be
cropped per object afterwards -- render cost per SCENE rather than per object.

Tilt. The simulator produces an incline by rotating GRAVITY and leaving the ground flat.
For the clip to show a tilted table rather than objects sliding on a level one, the whole
world is rotated back by the same angle: the table is rotated about Y through a pivot on
its surface, and the object poses were already rotated to match when they were exported.
+x tilts DOWN, which is the direction the simulator slides them.

Reads  LAB/scene_poses.json
  {key: {"tilt_deg": float, "objects": {name: [{"loc": [...], "mat": [[...]]}, ...]}}}
Writes LAB/sim_<key>/f####.png

Run: LAB=<dir> <blender> --background --python scripts/blender_render_scene.py
"""
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")

import bpy
import mathutils

sys.path.insert(0, str(REPO_ROOT))
from src.render.views import VIEWS  # noqa: E402

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
LAB = Path(os.environ["LAB"])
CITY = REPO / "assets" / "scenes" / "city"
CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
# The judge sees 544x448; a human watching the clip should not have to. RENDER_SCALE
# multiplies both dimensions for viewing-quality output without moving the camera.
_RS = float(os.environ.get("RENDER_SCALE", "1"))
GROUND_Z = 0.706
PIVOT = mathutils.Vector((-0.05, -0.06, GROUND_Z))   # matches DROP_XY in the sim


def clear():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(c):
            c.remove(b)


def world_hdri(path):
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(path)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def import_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        return None
    for o in bpy.data.objects:
        o.select_set(False)
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def main():
    poses = json.loads((LAB / "scene_poses.json").read_text())
    # A LAB may define its own object set (a bigger scene, different assets). Prefer it;
    # fall back to the shared staging config. Objects named in scene_poses.json but absent
    # here get parked out of frame, which is why a mismatch renders an empty table.
    _lab_cfg = LAB / "scene.json"
    scene_cfg = json.loads((_lab_cfg if _lab_cfg.exists() else SCENE / "scene.json").read_text())
    _missing = set(next(iter(poses.values()))["objects"]) - set(scene_cfg["objects"])
    if _missing:
        raise SystemExit(f"poses name objects absent from scene config: {sorted(_missing)}")
    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))
    # LOOK F. The floor is a SHADOW CATCHER, not a surface. As an opaque plane it was lit
    # by the HDRI to mid-grey and it occluded the environment's own ground, so the street
    # vanished below the horizon and the table floated above a grey slab -- in every render
    # this project has produced. Invisible to camera, still receiving contact shadow, the
    # HDRI ground shows through and the table stays grounded. This is scene context the
    # judge uses to read contact and orientation, not decoration.
    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.active_object.is_shadow_catcher = True

    table = import_gltf(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf")
    # A WIDER TABLE for multi-object scenes. 14 staged objects need 0.70 m of depth and
    # the stock table gives 0.706 with no margin, so they overhang the edge and topple off
    # for reasons that have nothing to do with their material. Scaling X/Y only keeps the
    # surface at GROUND_Z, which is what the simulator assumes.
    tsx, tsy = float(os.environ.get("TABLE_SX", "1")), float(os.environ.get("TABLE_SY", "1"))
    if tsx != 1.0 or tsy != 1.0:
        table.scale = (table.scale[0] * tsx, table.scale[1] * tsy, table.scale[2])
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()
    ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
    table.location.z += GROUND_Z - ztop
    table_home = table.location.copy()

    objs, home = {}, {}
    for name, o in scene_cfg["objects"].items():
        gl = list((REPO / "assets" / o["asset"]).glob("*.gltf"))
        if not gl:
            continue
        b = import_gltf(gl[0])
        if b is None:
            continue
        b.name = name; b.scale = (o["scale"],) * 3
        b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
        b.location = tuple(o["pos"]); objs[name] = b; home[name] = tuple(o["pos"])

    cams = {}
    for vn, vc in VIEWS.items():
        cd_ = bpy.data.cameras.new(f"c{vn}")
        co = bpy.data.objects.new(f"c{vn}", cd_)
        bpy.context.collection.objects.link(co)
        # CAM_PULL backs the camera away from its target along the same sight line, for
        # scenes wider than the one these views were framed for. Direction and target are
        # untouched, so the composition is preserved -- only the field of coverage grows.
        _eye = mathutils.Vector(vc["eye"]); _tgt = mathutils.Vector(vc["target"])
        _pull = float(os.environ.get("CAM_PULL", "1"))
        co.location = _tgt + (_eye - _tgt) * _pull
        lk = _tgt - co.location
        co.rotation_euler = lk.to_track_quat("-Z", "Y").to_euler()
        cd_.sensor_fit = "HORIZONTAL"; cd_.angle = math.radians(vc["fov_deg"])
        cams[vn] = co
    only = os.environ.get("VIEWS")
    if only:
        cams = {k: v for k, v in cams.items() if k in only.split(",")}
    bpy.context.scene.camera = cams[list(cams)[0]]

    sc = bpy.context.scene
    sc.render.engine = os.environ.get("ENGINE", "CYCLES")
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = int(os.environ.get("SAMPLES", 48))
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"; prefs.get_devices()
            for d in prefs.devices:
                d.use = (d.type == "CUDA")
            sc.cycles.device = "GPU"
        except Exception as e:
            print("GPU off:", e)
    sc.render.resolution_x = int(CAM["width"] * _RS)
    sc.render.resolution_y = int(CAM["height"] * _RS)
    sc.render.image_settings.file_format = "PNG"

    for key, info in poses.items():
      acting = info["objects"]
      n_frames = min(len(v) for v in acting.values())
      for vn, camobj in cams.items():
        bpy.context.scene.camera = camobj
        outdir = LAB / f"sim_{key}@{vn}"
        outdir.mkdir(parents=True, exist_ok=True)
        if len(list(outdir.glob("f*.png"))) == n_frames:
            print(f"  skip {key}@{vn}: already complete"); continue

        # tilt the table about Y through a pivot on its surface; +x goes down.
        # tilt_deg may be a scalar (fixed incline) or a per-frame list (ramp).
        tilt = info.get("tilt_deg", 0.0)
        tilt_seq = tilt if isinstance(tilt, list) else [float(tilt)] * n_frames
        table.rotation_mode = "XYZ"

        # anything not participating is moved out of frame rather than left floating
        for n, o in objs.items():
            if n not in acting:
                o.location = (home[n][0], home[n][1] + 3.0, home[n][2])
            else:
                o.rotation_mode = "QUATERNION"

        for t in range(n_frames):
            th = math.radians(float(tilt_seq[min(t, len(tilt_seq) - 1)]))
            R = mathutils.Matrix.Rotation(th, 4, "Y")
            table.rotation_euler = (0.0, th, 0.0)
            table.location = PIVOT + (R @ (table_home - PIVOT))
            for n, seq in acting.items():
                o = objs.get(n)
                if o is None:
                    continue
                p = seq[t]
                M = mathutils.Matrix([[p["mat"][r][c] for c in range(3)] for r in range(3)])
                o.rotation_quaternion = M.to_quaternion()
                o.location = mathutils.Vector(p["loc"])
            bpy.context.view_layer.update()
            sc.render.filepath = str(outdir / f"f{t:04d}.png")
            bpy.ops.render.render(write_still=True)

        # restore for the next key
        table.rotation_euler = (0.0, 0.0, 0.0); table.location = table_home
        for n, o in objs.items():
            o.rotation_mode = "XYZ"
            o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1],
                                scene_cfg["objects"][n]["rot_z"])
            o.location = home[n]
        print(f"  rendered {key}@{vn}: {n_frames} frames, {len(acting)} objects")


main()
