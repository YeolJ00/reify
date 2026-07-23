"""Render our simulated flag in a real city scene (Blender + PolyHaven CC0 assets).

Street ground + city HDRI (sky, lighting, background) + painted wooden bench +
street lamp + fire hydrant + a flagpole flying our cloth-sim flag. Replaces the
white-background renders with a photorealistic urban environment — a proper
initial frame I0 for the M4 i2v pipeline.

Run headless:
  CUDA_VISIBLE_DEVICES=<gpu> <blender> --background --python scripts/blender_city.py -- \
      --flag outputs/flag_frame.obj --out outputs/city_scene.png [--samples 96]
"""

import sys
from pathlib import Path

import bpy
import mathutils

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
CITY = REPO / "assets" / "scenes" / "city"


def argv_after_ddash():
    a = sys.argv
    return a[a.index("--") + 1:] if "--" in a else []


def parse(args):
    d = {"flag": str(REPO / "outputs" / "flag_frame.obj"),
         "out": str(REPO / "outputs" / "city_scene.png"), "samples": 96,
         "res": 900, "hdri": str(CITY / "pretville_street_2k.hdr")}
    for i in range(0, len(args) - 1, 2):
        k = args[i].lstrip("-")
        if k in d:
            d[k] = int(args[i + 1]) if k in ("samples", "res") else args[i + 1]
    return d


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(c):
            c.remove(b)


def set_world_hdri(path):
    w = bpy.data.worlds.new("city") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path)
    mapping = nt.nodes.new("ShaderNodeMapping")
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping.inputs["Rotation"].default_value[2] = 2.4  # rotate city so street recedes
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = 1.0


def import_gltf(folder, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(CITY / folder / f"{name}_2k.gltf"))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    # join into one, return it
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


def place(obj, loc, rot_z=0.0, scale=1.0):
    if obj is None:
        return
    obj.location = loc
    obj.rotation_euler = (0, 0, rot_z)
    obj.scale = (scale, scale, scale)
    # drop so the object's lowest point sits on z=loc[2]
    bpy.context.view_layer.update()
    zmin = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    obj.location.z += loc[2] - zmin


def street_ground():
    bpy.ops.mesh.primitive_plane_add(size=40)
    g = bpy.context.active_object
    mat = bpy.data.materials.new("asphalt")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.055, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    g.data.materials.append(mat)
    return g


def add_flag(obj_path):
    # our OBJ is Z-up (sim coords); keep it, don't let the importer rotate to Y-up
    bpy.ops.wm.obj_import(filepath=obj_path, up_axis="Z", forward_axis="Y")
    flag = bpy.context.selected_objects[0]
    # sim flag hangs in x-z, hoist at x~0; stand it on a pole at origin
    flag.location = (0, 0, 0)
    mat = bpy.data.materials.new("flag_fabric")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.12, 0.13, 1)  # red flag
    bsdf.inputs["Roughness"].default_value = 0.7
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.4
    flag.data.materials.clear()
    flag.data.materials.append(mat)
    for p in flag.data.polygons:
        p.use_smooth = True
    # flagpole
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=2.6, location=(0, 0, 1.3))
    pole = bpy.context.active_object
    pmat = bpy.data.materials.new("pole")
    pmat.use_nodes = True
    pb = pmat.node_tree.nodes["Principled BSDF"]
    pb.inputs["Base Color"].default_value = (0.4, 0.4, 0.42, 1)
    pb.inputs["Metallic"].default_value = 1.0
    pb.inputs["Roughness"].default_value = 0.35
    pole.data.materials.append(pmat)


def setup_camera_render(d):
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (3.6, -3.4, 1.75)
    look = mathutils.Vector((0.35, 0.1, 1.6)) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = d["samples"]
    sc.render.resolution_x = d["res"]
    sc.render.resolution_y = int(d["res"] * 0.72)
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = d["out"]
    # GPU
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = (dev.type == "CUDA")
        sc.cycles.device = "GPU"
    except Exception as e:
        print("GPU setup failed, using CPU:", e)


def main():
    d = parse(argv_after_ddash())
    clear_scene()
    set_world_hdri(d["hdri"])
    street_ground()
    place(import_gltf("painted_wooden_bench", "painted_wooden_bench"), (2.4, -1.1, 0), rot_z=-0.7)
    place(import_gltf("street_lamp_01", "street_lamp_01"), (-2.4, 1.4, 0), rot_z=0.4)
    place(import_gltf("fire_hydrant", "fire_hydrant"), (1.9, 1.3, 0), rot_z=1.0)
    add_flag(d["flag"])
    setup_camera_render(d)
    print(f"rendering -> {d['out']} ({d['samples']} samples) ...")
    bpy.ops.render.render(write_still=True)
    print("done")


main()
