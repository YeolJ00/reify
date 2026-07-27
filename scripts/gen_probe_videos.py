"""Generate the Cosmos3 video for each probe of the matrix experiment.

One initial frame per probe (rendered by blender_probe_i0.py), one prompt per probe
describing the intended excitation. Several seeds each so we can keep the most
physical take. Writes outputs/probes_i2v/vid_<probe>_seed<k>.npz

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_probe_videos.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.video.i2v import CosmosI2V  # noqa: E402

OUT = REPO / "outputs" / "probes_i2v"
PROMPTS = {
    "drop": "The red ball falls straight down under gravity onto the wooden tabletop, "
            "bounces, and settles. The blue ball stays exactly where it is.",
    # object-referenced directions: Cosmos follows "toward the blue ball" far more
    # reliably than "to the right" (which it often reversed)
    "push": "The red ball rolls steadily across the wooden tabletop in the direction of the "
            "blue ball but stops well before reaching it, slowing down from friction. "
            "The blue ball stays exactly where it is.",
    "collide": "The red ball rolls across the wooden tabletop toward the blue ball, hits the "
               "blue ball, and knocks the blue ball rolling away. The red ball slows down "
               "after the impact.",
}
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]


def main():
    backend = CosmosI2V()
    for probe, prompt in PROMPTS.items():
        if ONLY and probe not in ONLY:
            continue
        img = Image.open(OUT / f"I0_{probe}.png").convert("RGB")
        for seed in SEEDS:
            t0 = time.time()
            fr = backend.generate(img, prompt, num_frames=49, seed=seed, height=448, width=544)
            np.savez_compressed(OUT / f"vid_{probe}_seed{seed}.npz", frames=fr, prompt=prompt, seed=seed)
            print(f"{probe} seed{seed}: {fr.shape} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
