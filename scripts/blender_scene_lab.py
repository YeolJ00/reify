"""Stage the three DIFFERENT experiments on scene objects, for joint recovery.

One experiment cannot pin every parameter — our matrix measured which reveals which:
    drop     the object falls              -> restitution   (19.0 px effect vs 0.1 in a push)
    slide    it travels across the table   -> friction      (67.3 px vs 3.3 in a drop)
    collide  it strikes another object     -> mass ratio    (66.9 px vs 0.4 otherwise)
Solved together they constrain each other; solved separately each needs the others fixed
at a guess, and a wrong guess biases the answer that does get reported.

The pair used is the baseball and the apple: both small, compact and well tracked, and
the apple is light enough to be visibly knocked.

Run: CUDA_VISIBLE_DEVICES=<g> <blender> --background --python scripts/blender_scene_lab.py
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
SCENE = REPO / "outputs" / "scene"
OUT = SCENE / "lab"
CITY = REPO / "assets" / "scenes" / "city"

CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
GROUND_Z = 0.706
# The red apple is the MOVER and the white baseball the TARGET, not the other way round:
# in the first attempt the baseball tracked at 4-53% (white on a bright table, CoTracker
# lost it and followed things off-frame) while the apple tracked at 97-99%.
MOVER, TARGET = "apple", "baseball"

# experiment -> where the mover starts, where the target sits (None = leave it home)
LAYOUTS = {
    "drop":    {"mover": (-0.30, -0.02, 0.20), "target": None},   # z is a LIFT above rest
    "slide":   {"mover": (-0.42, -0.02, 0.0), "target": None},    # long clear runway
    # Close the gap. The first attempt put 31 cm of clear table between them and the
    # mover never arrived in the 2 s clip — the target did not move in ANY take. The
    # two-ball scene that did collide had them ~11 cm apart.
    "collide": {"mover": (-0.30, -0.02, 0.0), "target": (-0.10, -0.02, 0.0)},
}


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
    sc.render.engine = "CYCLES"; sc.cycles.samples = 72
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

    # clear the runway so the mover has an unobstructed path in slide/collide
    parked = {}
    for n in ("wooden_bowl", "book", "ceramic_vase", "rubber_duck", "brass_pot"):
        if n in objs:
            parked[n] = objs[n].location.copy()
            objs[n].location = (home[n][0], home[n][1] + 0.46, home[n][2])

    exp = {}
    for ename, lay in LAYOUTS.items():
        mv = objs[MOVER]; tg = objs[TARGET]
        mx, my, mlift = lay["mover"]
        mv.location = (mx, my, home[MOVER][2] + mlift)
        if lay["target"] is None:
            tg.location = home[TARGET]
        else:
            tx, ty, tl = lay["target"]
            tg.location = (tx, ty, home[TARGET][2] + tl)
        bpy.context.view_layer.update()
        sc.render.filepath = str(OUT / f"I0_{ename}.png")
        bpy.ops.render.render(write_still=True)
        exp[ename] = {"mover": MOVER, "target": TARGET,
                      "mover_pos": list(mv.location), "target_pos": list(tg.location),
                      "mover_lift": mlift}
        print(f"  staged {ename:8s} mover at {tuple(round(v,3) for v in mv.location)}")
        mv.location = home[MOVER]; tg.location = home[TARGET]

    (OUT / "lab.json").write_text(json.dumps(
        {"camera": CAM, "ground_z": GROUND_Z, "mover": MOVER, "target": TARGET,
         "experiments": exp,
         "assets": {n: cfg["objects"][n] for n in (MOVER, TARGET)}}, indent=2))
    print("wrote", OUT / "lab.json")


main()
