"""Generate the expanded lab: 7 objects x (3 drop heights + slide) x N seeds.

Collide is deliberately absent -- it yielded ~24% usable against ~80% for drop and slide,
so the same GPU hours buy roughly three times the data here.

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... [SEEDS=0,1,2,3,4,5] \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_expand.py
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

OUT = REPO / "outputs" / "scene" / "expand"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5").split(",")]

NOUN = {"apple": "red apple", "baseball": "white baseball", "brass_pot": "copper pot",
        "ceramic_vase": "white ceramic vase", "rubber_duck": "yellow rubber duck",
        "wooden_bowl": "wooden bowl", "book": "hardcover book"}


def prompt_for(e):
    s = NOUN.get(e["subject"], e["subject"].replace("_", " "))
    if e["kind"] == "drop":
        return (f"The {s} falls onto the wooden tabletop, hits it and visibly bounces "
                f"back up, then comes to rest. Nothing else moves.")
    return (f"The {s} is already sliding and travels across the wooden tabletop, "
            f"gradually slowing down from friction, and stops on its own. "
            f"Nothing else moves.")


def main():
    cfg = json.loads((OUT / "lab.json").read_text())
    backend = CosmosI2V()
    todo = sorted(cfg["experiments"])
    n = len(todo) * len(SEEDS)
    print(f"{len(todo)} experiments x {len(SEEDS)} seeds = {n} clips", flush=True)
    done = 0
    for key in todo:
        e = cfg["experiments"][key]
        img = Image.open(OUT / f"I0_{key}.png").convert("RGB")
        pr = prompt_for(e)
        for seed in SEEDS:
            dst = OUT / f"vid_{key}_seed{seed}.npz"
            if dst.exists():
                done += 1
                continue
            t0 = time.time()
            fr = backend.generate(img, pr, num_frames=49, seed=seed,
                                  height=448, width=544)
            np.savez_compressed(dst, frames=fr, prompt=pr, seed=seed)
            done += 1
            print(f"[{done}/{n}] {key}_seed{seed}: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
