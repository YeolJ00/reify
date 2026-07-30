"""Generate the screening clips: same motion asked for, four different stagings.

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... [SEEDS=0,1,2] \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_screen.py
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

OUT = REPO / "outputs" / "scene" / "screen"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]

NOUN = {"apple": "red apple", "baseball": "white baseball", "brass_pot": "copper pot",
        "ceramic_vase": "white ceramic vase", "rubber_duck": "yellow rubber duck"}

# The same requested outcome in every case -- the object ends up having slid to the right
# and stopped. Only the staging, and for 'urgent' only the wording, differs.
PROMPT = {
    "rest": ("The {s} is already sliding and travels across the wooden tabletop, "
             "gradually slowing down from friction, and stops on its own. "
             "Nothing else moves."),
    "urgent": ("The {s} is already moving fast. It shoots to the right across the "
               "wooden tabletop, decelerating as it goes, and comes to a stop near the "
               "right edge of the table. It definitely moves a long way. "
               "Nothing else moves."),
    "airborne": ("The {s} falls, lands on the wooden tabletop, and slides to the right "
                 "before coming to a stop. Nothing else moves."),
    "tipped": ("The {s} topples over onto the wooden tabletop and slides to the right "
               "before coming to a stop. Nothing else moves."),
}


def main():
    cfg = json.loads((OUT / "lab.json").read_text())
    backend = CosmosI2V()
    todo = sorted(cfg["experiments"])
    print(f"{len(todo)} variants x {len(SEEDS)} seeds = {len(todo)*len(SEEDS)} clips",
          flush=True)
    for key in todo:
        e = cfg["experiments"][key]
        img = Image.open(OUT / f"I0_{key}.png").convert("RGB")
        s = NOUN.get(e["subject"], e["subject"].replace("_", " "))
        pr = PROMPT[e["variant"]].format(s=s)
        for seed in SEEDS:
            dst = OUT / f"vid_{key}_seed{seed}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            fr = backend.generate(img, pr, num_frames=49, seed=seed,
                                  height=448, width=544)
            np.savez_compressed(dst, frames=fr, prompt=pr, seed=seed)
            print(f"{key}_seed{seed}: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
