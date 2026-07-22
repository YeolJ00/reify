"""i2v backends for M4 stage 2 (architecture step 3): one interface, three models.

Backends (all weights in HF_HOME=/home/nas5/jooyeolyun/hf_cache):
  wan5b    Wan-AI/Wan2.2-TI2V-5B-Diffusers        via WanImageToVideoPipeline
  hunyuan  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v
                                                   via HunyuanVideo15ImageToVideoPipeline
  cosmos3  nvidia/Cosmos3-Nano                     via a running vLLM-Omni server (HTTP)

Run in the `video` conda env (torch), NOT the `warp` env. A6000 is sm_86
(Ampere): bf16 only, no FP8 paths.

The generated video is a MOTION PRIOR: we only extract tracks from it, so we
generate at modest resolution and prioritize (a) first-frame faithfulness and
(b) a static camera. STATIC_CAMERA_SUFFIX is appended to every prompt.
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/nas5/jooyeolyun/hf_cache")

import numpy as np

STATIC_CAMERA_SUFFIX = (
    " The camera is completely static, locked on a tripod, no camera motion, no zoom, "
    "no pan, fixed framing for the entire video."
)

NEGATIVE_PROMPT = (
    "camera motion, zoom, pan, dolly, shaky footage, scene cut, fade, morphing objects, "
    "appearing objects, disappearing objects, text, watermark, blurry, low quality"
)

WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
HUNYUAN_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
COSMOS_REPO = "nvidia/Cosmos3-Nano"


def _to_uint8_frames(video) -> np.ndarray:
    """diffusers output (list of PIL or np float) -> (F,H,W,3) uint8."""
    frames = []
    for fr in video:
        a = np.asarray(fr)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        frames.append(a[..., :3])
    return np.stack(frames)


class WanI2V:
    """Wan2.2 TI2V-5B. ~24 fps native, 720p max; we default to 704x704-ish area."""

    def __init__(self, device="cuda"):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        self.torch = torch
        vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae",
                                               torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_REPO, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device)
        self.fps = 24

    def generate(self, image, prompt, num_frames=49, seed=0, height=480, width=480,
                 steps=35, guidance=5.0):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            image=image, prompt=prompt + STATIC_CAMERA_SUFFIX,
            negative_prompt=NEGATIVE_PROMPT, height=height, width=width,
            num_frames=num_frames, num_inference_steps=steps,
            guidance_scale=guidance, generator=g,
        )
        return _to_uint8_frames(out.frames[0])


class HunyuanI2V:
    """HunyuanVideo 1.5 480p i2v — the physics/cloth-motion candidate."""

    def __init__(self, device="cuda"):
        import torch
        from diffusers import HunyuanVideo15ImageToVideoPipeline

        self.torch = torch
        self.pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            HUNYUAN_REPO, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device)
        self.fps = 24

    def generate(self, image, prompt, num_frames=49, seed=0, height=480, width=480,
                 steps=30, guidance=6.0):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            image=image, prompt=prompt + STATIC_CAMERA_SUFFIX,
            negative_prompt=NEGATIVE_PROMPT, height=height, width=width,
            num_frames=num_frames, num_inference_steps=steps,
            guidance_scale=guidance, generator=g,
        )
        return _to_uint8_frames(out.frames[0])


class Cosmos3I2V:
    """Cosmos3-Nano through a vLLM-Omni server (see model card).

    Start the server first (video env):
      vllm serve nvidia/Cosmos3-Nano --omni ...   # see docs/PROJECT_LOG.md
    then point base_url at it. This client posts an i2v request and polls.
    """

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.fps = 24

    def generate(self, image, prompt, num_frames=49, seed=0, **kw):
        import base64
        import io

        import requests

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = requests.post(
            f"{self.base_url}/v1/video/generations",
            json={
                "model": COSMOS_REPO,
                "prompt": prompt + STATIC_CAMERA_SUFFIX,
                "image": f"data:image/png;base64,{b64}",
                "num_frames": num_frames,
                "seed": seed,
            },
            timeout=1800,
        )
        r.raise_for_status()
        raise NotImplementedError(
            "decode of the server response is finalized in the bake-off milestone; "
            "weights + server recipe are ready, see docs/PROJECT_LOG.md"
        )


BACKENDS = {"wan5b": WanI2V, "hunyuan": HunyuanI2V, "cosmos3": Cosmos3I2V}


def save_video(frames: np.ndarray, path: Path, fps: int = 24):
    import imageio.v3 as iio

    iio.imwrite(path, frames, fps=fps, codec="libx264", quality=8)
