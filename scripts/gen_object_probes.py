"""Generate the Cosmos3 drop-probe video for each object of the authored scene.

One video per object, conditioned on that object's own lifted initial frame, so the
motion we fit belongs to that object in that scene. Several seeds each — most generated
takes fail the physical-validity screen, so we need candidates to choose from.

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_object_probes.py
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

OUT = REPO / "outputs" / "scene" / "probes"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]

# how to describe each object to the video model
NOUN = {"baseball": "white baseball", "apple": "red apple", "brass_pot": "copper pot",
        "ceramic_vase": "white ceramic vase", "rubber_duck": "yellow rubber duck"}


def main():
    cfg = json.loads((OUT / "probes.json").read_text())
    backend = CosmosI2V()
    for name in cfg["probes"]:
        if ONLY and name not in ONLY:
            continue
        noun = NOUN.get(name, name.replace("_", " "))
        prompt = (f"The {noun} falls from a height and hits the wooden tabletop hard, "
                  f"visibly bounces back up, bounces again lower, and finally comes to rest. "
                  f"Every other object on the table stays exactly where it is.")
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
