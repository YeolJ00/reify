"""Camera viewpoints, shared by the Blender render script and the crop code.

These live here rather than in blender_render_scene.py because the crop path runs in the
warp env and importing that script pulls in bpy, which is only available inside Blender.

Several views from ONE simulation. The sim is the expensive part of the ramp probe -- 48
sequential rollouts per object -- and is reused entirely; only pixels are re-rendered.

View "a" is the default three-quarter angle, inherited from the generated-video phase where
it had to match a Cosmos initial frame. It is a poor angle for reading a SLIP ANGLE, which
is why "b" exists: side-on and perpendicular to the slope, the incline is a clean line and
motion along it is unambiguous.
"""
VIEWS = {
    "a": {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
          "width": 544, "height": 448},
    "b": {"eye": [-0.05, -1.35, 0.92], "target": [-0.05, 0.00, 0.78], "fov_deg": 40,
          "width": 544, "height": 448},
    "c": {"eye": [0.50, -0.62, 1.62], "target": [-0.05, 0.00, 0.74], "fov_deg": 48,
          "width": 544, "height": 448},
}
