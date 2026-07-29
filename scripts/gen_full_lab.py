"""Generate the Cosmos clips for the full scene-wide lab.

Several seeds per experiment on purpose: the model is not repeatable (the same object
under the same prompt rebounds anywhere from 0% to 39% depending only on the seed), so a
single clip is a sample, not a measurement. The fitter treats agreement across seeds as
the uncertainty, which needs more than one take to be meaningful.

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... [SEEDS=0,1,2] [ONLY=apple_drop] \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_full_lab.py
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

OUT = REPO / "outputs" / "scene" / "fulllab"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]

NOUN = {"apple": "red apple", "baseball": "white baseball", "brass_pot": "copper pot",
        "ceramic_vase": "white ceramic vase", "rubber_duck": "yellow rubber duck"}


def prompt_for(e):
    s = NOUN.get(e["subject"], e["subject"].replace("_", " "))
    if e["kind"] == "drop":
        return (f"The {s} falls onto the wooden tabletop, hits it and visibly bounces back "
                f"up, then comes to rest. Nothing else moves.")
    if e["kind"] == "slide":
        return (f"The {s} is already sliding and travels across the wooden tabletop, "
                f"gradually slowing down from friction, and stops on its own. "
                f"Nothing else moves.")
    p = NOUN.get(e["partner"], (e["partner"] or "object").replace("_", " "))
    return (f"The {s} slides a short distance across the wooden tabletop into the {p} right "
            f"next to it, strikes it, and the {p} is knocked away and rolls off. "
            f"The {s} slows down after the impact.")


def main():
    cfg = json.loads((OUT / "lab.json").read_text())
    backend = CosmosI2V()
    todo = [k for k in cfg["experiments"] if not ONLY or k in ONLY]
    print(f"{len(todo)} experiments x {len(SEEDS)} seeds = {len(todo)*len(SEEDS)} clips")
    for key in todo:
        e = cfg["experiments"][key]
        img = Image.open(OUT / f"I0_{key}.png").convert("RGB")
        pr = prompt_for(e)
        for seed in SEEDS:
            dst = OUT / f"vid_{key}_seed{seed}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            fr = backend.generate(img, pr, num_frames=49, seed=seed, height=448, width=544)
            np.savez_compressed(dst, frames=fr, prompt=pr, seed=seed)
            print(f"{key}_seed{seed}: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
