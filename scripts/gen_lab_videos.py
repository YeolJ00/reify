"""Generate the Cosmos video for each of the three lab experiments.

Prompt design follows what we learned the hard way:
  * say what should happen AFTER release, not that something pushes it — our simulator
    models one impulse at t=0 and then only gravity/contact/friction, so a clip where the
    object keeps accelerating is unexplainable by construction;
  * ask for the effect we intend to measure (a visible bounce in the drop), because a
    prompt that says "lands and settles" instructs the model not to bounce;
  * reference the other object by name for direction — Cosmos reverses plain "to the right".

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_lab_videos.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.video.i2v import CosmosI2V  # noqa: E402

OUT = REPO / "outputs" / "scene" / "lab"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]

PROMPTS = {
    "drop": "The white baseball falls onto the wooden tabletop, hits it and visibly bounces "
            "back up, bounces again lower, then comes to rest. Nothing else moves.",
    "slide": "The white baseball is already rolling and travels across the wooden tabletop "
             "toward the red apple, gradually slowing down from friction, and stops on its "
             "own well before reaching it. Nothing else moves.",
    "collide": "The white baseball rolls across the wooden tabletop into the red apple, "
               "strikes it, and knocks the apple rolling away. The baseball slows after "
               "the impact. Nothing else moves.",
}


def main():
    backend = CosmosI2V()
    for name, prompt in PROMPTS.items():
        img = Image.open(OUT / f"I0_{name}.png").convert("RGB")
        for seed in SEEDS:
            dst = OUT / f"vid_{name}_seed{seed}.npz"
            if dst.exists():
                print(f"skip {name} seed{seed}"); continue
            t0 = time.time()
            fr = backend.generate(img, prompt, num_frames=49, seed=seed, height=448, width=544)
            np.savez_compressed(dst, frames=fr, prompt=prompt, seed=seed)
            print(f"{name} seed{seed}: {fr.shape} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
