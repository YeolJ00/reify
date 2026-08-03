"""Generate the prompting A/B: 2 objects x 4 arms x N seeds, drop_mid staging.

Arms isolate the two documented things we were not doing -- the JSON prompt structure the
model card asks for, and the negative prompt it ships -- so their effects can be separated
rather than confounded.

Run (cosmos env): CUDA_VISIBLE_DEVICES=<g> HF_HOME=... [SEEDS=0,1,2,3,4,5] \
    /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/gen_ab.py
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

from scripts.ab_prompt import ARMS, negative_prompt, plain, structured  # noqa: E402
from src.video.i2v import CosmosI2V  # noqa: E402

EXP = REPO / "outputs" / "scene" / "expand"
OUT = REPO / "outputs" / "scene" / "ab"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5").split(",")]
SUBJECTS = ["ceramic_vase", "brass_pot"]      # one that measures, one that degrades


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    neg = negative_prompt()
    backend = CosmosI2V()
    n = len(SUBJECTS) * len(ARMS) * len(SEEDS)
    print(f"{len(SUBJECTS)} objects x {len(ARMS)} arms x {len(SEEDS)} seeds = {n} clips",
          flush=True)
    done = 0
    for subj in SUBJECTS:
        img = Image.open(EXP / f"I0_{subj}_drop_mid.png").convert("RGB")
        for arm, cfg in ARMS.items():
            pr = structured(subj) if cfg["structured"] else plain(subj)
            npx = neg if cfg["negative"] else None
            for seed in SEEDS:
                dst = OUT / f"vid_{subj}_{arm}_seed{seed}.npz"
                if dst.exists():
                    done += 1
                    continue
                t0 = time.time()
                fr = backend.generate(img, pr, num_frames=49, seed=seed,
                                      height=448, width=544,
                                      negative_prompt=npx,
                                      raw_json=cfg["structured"])
                np.savez_compressed(dst, frames=fr, arm=arm, seed=seed)
                done += 1
                print(f"[{done}/{n}] {subj}_{arm}_seed{seed}: {time.time()-t0:.0f}s",
                      flush=True)
    (OUT / "arms.json").write_text(json.dumps(
        {"subjects": SUBJECTS, "arms": {k: v for k, v in ARMS.items()},
         "seeds": SEEDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
