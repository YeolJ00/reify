"""Render one frame of the probe scene under several looks, to choose lighting and backdrop.

The current look was inherited from the generated-video phase, where the scene had to match
a Cosmos initial frame. Nothing constrains it now -- we are not matching footage -- so the
backdrop and lighting are free variables and should be chosen for legibility instead.

Each variant renders the SAME frame with the same camera and objects, so the only
difference is the look. Written as stills because lighting is judged from one frame and a
still costs 1/40th of a clip.

Run: LAB=<dir> <blender> --background --python scripts/blender_look_dev.py
"""
import json
import math
import os
from pathlib import Path

import bpy
import mathutils

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
LAB = Path(os.environ["LAB"])
OUT = LAB / "look"
CITY = REPO / "assets" / "scenes" / "city" / "pretville_street_2k.hdr"
STUDIO = REPO / "assets" / "scenes" / "studio_small_03_2k.hdr"
CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 640, "height": 512}
GROUND_Z = 0.706
FRAME = 30           # mid-clip, objects in motion

# name -> (hdri, strength, backdrop, key light watts, floor colour)
LOOKS = {
    "A_current":   (CITY, 1.0, "none", 0, (.05, .05, .06)),
    "B_studio":    (STUDIO, 1.0, "none", 0, (.05, .05, .06)),
    "C_studio_bg": (STUDIO, 0.6, "seamless", 0, (.18, .17, .16)),
    "D_key_light": (STUDIO, 0.35, "seamless", 320, (.18, .17, .16)),
    "E_soft_dark": (STUDIO, 0.25, "seamless", 180, (.07, .07, .08)),
}


def clear():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights):
        for b in list(c):
            c.remove(b)


def world_hdri(path, strength):
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(path))
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
    scene_cfg = json.loads((SCENE / "scene.json").read_text())
    key = sorted(poses)[-1]
    info = poses[key]
    OUT.mkdir(parents=True, exist_ok=True)

    for name, (hdri, strength, backdrop, watts, floor_col) in LOOKS.items():
        clear()
        world_hdri(hdri, strength)

        # floor, and optionally a seamless curved backdrop that removes the street cutout
        bpy.ops.mesh.primitive_plane_add(size=40)
        fl = bpy.context.active_object
        m = bpy.data.materials.new("floor"); m.use_nodes = True
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (*floor_col, 1)
        p.inputs["Roughness"].default_value = 0.85
        fl.data.materials.append(m)
        if backdrop == "seamless":
            # a large curved wall behind the table: the HDRI's street imagery reads as a
            # cutout across the top of frame, which is scene clutter with no physical role
            # A closed cylinder around the camera blocks the HDRI completely -- variant C
            # rendered black for exactly that reason. Use an OPEN curved wall placed
            # BEHIND the subject instead, so the environment still lights the scene.
            bpy.ops.mesh.primitive_plane_add(size=1)
            cyc = bpy.context.active_object
            cyc.scale = (7.0, 1.0, 4.5)
            cyc.rotation_euler = (math.radians(90), 0, 0)
            cyc.location = (0.0, 3.2, 1.4)
            mb = bpy.data.materials.new("bd"); mb.use_nodes = True
            pb = mb.node_tree.nodes["Principled BSDF"]
            pb.inputs["Base Color"].default_value = (0.22, 0.21, 0.20, 1)
            pb.inputs["Roughness"].default_value = 0.95
            cyc.data.materials.append(mb)

        table = import_gltf(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf")
        bpy.context.view_layer.update()
        ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
        table.location.z += GROUND_Z - ztop

        objs = {}
        for nm, o in scene_cfg["objects"].items():
            gl = list((REPO / "assets" / o["asset"]).glob("*.gltf"))
            if not gl:
                continue
            b = import_gltf(gl[0])
            if b is None:
                continue
            b.name = nm; b.scale = (o["scale"],) * 3
            b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
            b.location = tuple(o["pos"]); objs[nm] = b
            if nm not in info["objects"]:
                b.location = (o["pos"][0], o["pos"][1] + 3.0, o["pos"][2])

        if watts:
            ld = bpy.data.lights.new("key", type="AREA"); ld.energy = watts; ld.size = 1.6
            lo = bpy.data.objects.new("key", ld)
            bpy.context.collection.objects.link(lo)
            lo.location = (1.4, -1.3, 2.3)
            d = mathutils.Vector((0, 0, GROUND_Z)) - lo.location
            lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            fd = bpy.data.lights.new("fill", type="AREA"); fd.energy = watts * 0.3
            fd.size = 2.5
            fo = bpy.data.objects.new("fill", fd)
            bpy.context.collection.objects.link(fo)
            fo.location = (-1.6, -0.9, 1.9)
            d2 = mathutils.Vector((0, 0, GROUND_Z)) - fo.location
            fo.rotation_euler = d2.to_track_quat("-Z", "Y").to_euler()

        cd = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cd)
        bpy.context.collection.objects.link(cam)
        cam.location = mathutils.Vector(CAM["eye"])
        look = mathutils.Vector(CAM["target"]) - cam.location
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        cd.sensor_fit = "HORIZONTAL"; cd.angle = math.radians(CAM["fov_deg"])
        bpy.context.scene.camera = cam

        sc = bpy.context.scene
        sc.render.engine = "CYCLES"; sc.cycles.samples = 96
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

        tilt = info.get("tilt_deg", 0.0)
        tseq = tilt if isinstance(tilt, list) else [tilt]
        th = math.radians(float(tseq[min(FRAME, len(tseq) - 1)]))
        R = mathutils.Matrix.Rotation(th, 4, "Y")
        piv = mathutils.Vector((-0.05, -0.06, GROUND_Z))
        table.rotation_mode = "XYZ"; table.rotation_euler = (0, th, 0)
        thome = mathutils.Vector((table.location.x, table.location.y, table.location.z))
        table.location = piv + (R @ (thome - piv))
        for nm, seq in info["objects"].items():
            o = objs.get(nm)
            if o is None:
                continue
            pp = seq[min(FRAME, len(seq) - 1)]
            M = mathutils.Matrix([[pp["mat"][r][c] for c in range(3)] for r in range(3)])
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = M.to_quaternion()
            o.location = mathutils.Vector(pp["loc"])
        bpy.context.view_layer.update()
        sc.render.filepath = str(OUT / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"  rendered {name}")


main()
