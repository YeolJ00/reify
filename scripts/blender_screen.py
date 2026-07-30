"""Stage a generator SCREEN: does the initial frame have to imply the motion?

We asked Cosmos to slide objects that its own initial frame showed at rest on a table,
and it mostly refused -- correctly, arguably, since a resting object staying at rest is
the plausible continuation. This screens four stagings of the SAME motion to separate
"the prompt was too weak" from "the image depicted a static equilibrium":

    rest      object at rest on the table          EQUILIBRIUM (control: what we shipped)
    urgent    identical frame, forceful prompt     EQUILIBRIUM (isolates prompt from image)
    airborne  object 12 cm above the table         NOT in equilibrium (must fall)
    tipped    object leaning past its balance      NOT in equilibrium (must topple)

The split that matters is the last column. A ramp prop was tried first and abandoned:
getting the object to rest convincingly ON the incline needed contact geometry we do not
have here, and a floating object above a plank tests nothing. Tipping the object itself
depicts non-equilibrium with no second prop and no contact to get wrong.

Only the staging differs between rest and urgent -- nothing at all, in fact; they share
an initial frame, so any difference is purely the prompt. Written into its own lab dir in
the same schema as the main lab, so prepare_lab.py seeds/track work unchanged.

Run: <blender> --background --python scripts/blender_screen.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
OUT = SCENE / "screen"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
GROUND_Z = 0.706
LANE_Y = -0.02
START_X = -0.34
PARK_Y = 0.46
SUBJECTS = ["ceramic_vase", "rubber_duck", "baseball"]   # 2 that never moved + 1 control
RAMP_TILT_DEG = 22.0
RAMP_LEN = 0.22
RAMP_MID_Z = 0.030


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
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SCENE / "scene.json").read_text())
    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))
    bpy.ops.mesh.primitive_plane_add(size=40)
    fl = bpy.context.active_object
    m = bpy.data.materials.new("floor"); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.05, .05, .06, 1)
    fl.data.materials.append(m)

    table = import_gltf(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf")
    bpy.context.view_layer.update()
    ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
    table.location.z += GROUND_Z - ztop

    # a plank, hidden until the ramp variant needs it
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ramp = bpy.context.active_object; ramp.name = "ramp"
    # a thin plank did not read as an incline at all -- it looked like the object was
    # balanced on a stick, which would have produced a misleading negative
    ramp.scale = (RAMP_LEN, 0.11, 0.022)
    ramp.rotation_euler = (0.0, math.radians(-RAMP_TILT_DEG), 0.0)
    rm = bpy.data.materials.new("ramp"); rm.use_nodes = True
    rm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.32, .21, .12, 1)
    ramp.data.materials.append(rm)
    ramp.location = (START_X + 0.06, LANE_Y, GROUND_Z + RAMP_MID_Z)
    # the object goes at the RAISED end so gravity clearly points down the slope
    RAMP_H = RAMP_MID_Z + 0.5 * RAMP_LEN * math.sin(math.radians(RAMP_TILT_DEG))

    objs, home = {}, {}
    for name, o in cfg["objects"].items():
        gl = list((REPO / "assets" / o["asset"]).glob("*.gltf"))
        if not gl:
            continue
        b = import_gltf(gl[0])
        if b is None:
            continue
        b.name = name; b.scale = (o["scale"],) * 3
        b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
        b.location = tuple(o["pos"]); objs[name] = b; home[name] = tuple(o["pos"])

    cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(CAM["eye"])
    look = mathutils.Vector(CAM["target"]) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_d.sensor_fit = "HORIZONTAL"; cam_d.angle = math.radians(CAM["fov_deg"])
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = 64
    sc.render.resolution_x = CAM["width"]; sc.render.resolution_y = CAM["height"]
    sc.render.image_settings.file_format = "PNG"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"; prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
        sc.cycles.device = "GPU"
    except Exception as e:
        print("GPU off:", e)

    exps = {}
    for subj in SUBJECTS:
        if subj not in objs:
            print("missing", subj); continue
        for n, o in objs.items():
            if n != subj:
                o.location = (home[n][0], home[n][1] + PARK_Y, home[n][2])
        s = objs[subj]
        base_rot = tuple(s.rotation_euler)
        for variant, lift, tilt in (("rest", 0.0, 0.0), ("airborne", 0.12, 0.0),
                                    ("tipped", 0.012, 34.0)):
            ramp.hide_render = True
            s.rotation_euler = (base_rot[0], base_rot[1] + math.radians(tilt), base_rot[2])
            s.location = (START_X, LANE_Y, home[subj][2] + lift)
            bpy.context.view_layer.update()
            key = f"{subj}_{variant}"
            sc.render.filepath = str(OUT / f"I0_{key}.png")
            bpy.ops.render.render(write_still=True)
            exps[key] = {"subject": subj, "kind": "slide", "partner": None,
                         "subject_pos": list(s.location), "partner_pos": None,
                         "lift": lift, "variant": variant}
            print(f"  staged {key}")
        s.rotation_euler = base_rot
        for n in objs:
            objs[n].location = home[n]

    # 'urgent' shares the rest frame exactly: the only difference is the prompt, which
    # makes the prompt-vs-image question a controlled comparison rather than a guess
    for subj in SUBJECTS:
        if f"{subj}_rest" not in exps:
            continue
        src = OUT / f"I0_{subj}_rest.png"
        dst = OUT / f"I0_{subj}_urgent.png"
        dst.write_bytes(src.read_bytes())
        e = dict(exps[f"{subj}_rest"]); e["variant"] = "urgent"
        exps[f"{subj}_urgent"] = e
        print(f"  staged {subj}_urgent (same frame as rest)")

    (OUT / "lab.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "reference": None,
         "experiments": exps, "assets": cfg["objects"]}, indent=2))
    print(f"wrote {OUT}/lab.json  ({len(exps)} variants)")


main()
