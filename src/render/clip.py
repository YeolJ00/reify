"""J0: turn a cloth rollout into a short mp4 for the judge to watch.

Design notes that matter for what the judge sees:

* Rendered with `environment="default"`, not "studio": measured cloth-vs-background
  contrast is 80/255 against 39/255, twice as legible for a judge that has to read shape
  and motion. `log_mesh`'s `color` and `roughness` arguments were measured to have NO
  effect on the output (identical pixels for (0.86,0.86,0.90)/0.75 and (0.95,0.90,0.80)/
  0.35), so they are not used to carry material appearance -- which is fine here, since
  the judge should be reading motion rather than colour.
* Fixed camera, no motion, so differences between clips are the cloth's motion and nothing
  else.
* Deterministic: the same theta and seed produce the same frames, and clips are cached by
  hash(theta, scene, seed) so re-scoring never re-renders.

Cosmos-Reason samples video at 4 fps, so a clip needs to be long enough to contain several
sampled frames of actual motion -- a 1 s clip at 4 fps is 4 samples, which is not a motion
cue. Default here is ~3 s.
"""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def theta_key(theta: dict, scene: str, seed: int) -> str:
    """Stable hash for the render cache."""
    blob = json.dumps({"theta": {k: float(v) for k, v in sorted(theta.items())},
                       "scene": scene, "seed": int(seed)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def camera_angles(eye, target):
    import math
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    return (math.degrees(math.asin(float(np.clip(f[2], -1, 1)))),
            math.degrees(math.atan2(f[1], f[0])))


def render_frames(sim, eye, target, width=640, height=480, every=1,
                  environment="default"):
    """Render frame-boundary states of a completed rollout. Returns (T, H, W, 3) uint8."""
    import warp as wp
    import newton.viewer as V
    from PIL import Image

    states = sim.frame_states()[::every]
    tri = np.asarray(sim.model.tri_indices.numpy(), np.int32).ravel()
    pitch, yaw = camera_angles(eye, target)

    viewer = V.ViewerRTX(width=width, height=height, headless=True, up_axis="Z",
                         environment=environment)
    viewer.set_camera(wp.vec3(*[float(x) for x in eye]), pitch, yaw)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, st in enumerate(states):
            q = st.particle_q.numpy().astype(np.float32)
            viewer.begin_frame(i / 24.0)
            viewer.log_mesh("cloth", wp.array(q, dtype=wp.vec3),
                            wp.array(tri, dtype=wp.int32), backface_culling=False)
            viewer.end_frame()
            p = os.path.join(td, f"f{i:04d}.png")
            viewer.save_screenshot(p)
            out.append(np.asarray(Image.open(p).convert("RGB")))
    viewer.close()
    return np.stack(out)


def write_mp4(frames, path, fps=24):
    """Encode with ffmpeg via imageio if available, else OpenCV."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                               macro_block_size=1)
        for f in frames:
            w.append_data(f)
        w.close()
        return path
    except Exception:
        import cv2
        h, wd = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return path


def probe(path):
    """What actually landed on disk. Uses imageio rather than ffprobe, which is not
    installed here and silently turned every probe into an error string."""
    try:
        import imageio.v2 as imageio
        rd = imageio.get_reader(str(path))
        meta = rd.get_meta_data()
        n = rd.count_frames()
        rd.close()
        return {"frames": int(n), "fps": float(meta.get("fps", 0)),
                "size": "x".join(str(v) for v in meta.get("size", ()))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:60]}"}
