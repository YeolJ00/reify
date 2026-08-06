"""Run Cosmos3-Super's reasoner on the exact clips Nano was scored on, and compare.

Fitting it. transformers already loads ONLY the reasoner: its
_COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS list discards the generation expert (moe_gen),
the diffusion projections, the time embedder and the sound tower. That leaves ~30.5B
parameters of the 64B checkpoint -- but ~61 GB in bf16, which still does not fit a 48 GB
card, so this loads int8 (~31 GB). lm_head is excluded from quantisation because the score
is a logit difference read at one position, and that is exactly where precision matters;
the head then runs in fp32 as it does for Nano.

The comparison is like-for-like: same clips, same three prompts, same forced (A)/(B)
logit margin, same fp32 head. Only the model changes.

The decisive numbers are not the raw scores -- an int8 model has its own offset -- but:
  1. does it separate AUTHORED VIOLATIONS from their own valid twins, which Nano did not
     (Nano scored a 2.2x-swelling object HIGHER than real physics);
  2. is its score still rank-identical to motion magnitude, which for Nano was rho = +1.000
     across conditions.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=6 .../envs/cosmos/bin/python scripts/super_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

PH = REPO / "outputs" / "judge" / "phys"
G0 = REPO / "outputs" / "judge" / "g0c"
OUT = REPO / "outputs" / "judge" / "super"
MODEL = "nvidia/Cosmos3-Super"


def main():
    from src.judge.plausibility import PlausibilityJudge, QUESTION
    from src.render.motion_budget import sample_frames

    OUT.mkdir(parents=True, exist_ok=True)
    VARIANTS = {
        "plausible": QUESTION,
        "natural": ("How natural does the motion in this video look, compared to how this "
                    "object would move in the real world? Assume the normal laws of "
                    "physics.\nYour answer should be based on the events in the video and "
                    "ignore the quality of the simulation engine.\n(A) Natural\n"
                    "(B) Unnatural"),
        "material": ("Is the way this object moves consistent with what it is made of? "
                     "Consider its weight, hardness and how much such a material should "
                     "bounce. Assume the normal laws of physics.\nYour answer should be "
                     "based on the events in the video and ignore the quality of the "
                     "simulation engine.\n(A) Consistent\n(B) Inconsistent"),
    }

    import time
    import torch
    t0 = time.time()
    j = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"loaded in {time.time()-t0:.0f}s | "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")
    n_par = sum(p.numel() for p in j.model.parameters())
    print(f"reasoner parameters actually instantiated: {n_par/1e9:.1f}B "
          f"(checkpoint is ~64B including the generation expert)")

    rows = []
    # --- authored violations, the decisive set
    for p in sorted(PH.glob("*.mp4")):
        base, cond = p.stem.rsplit("__", 1)
        v = sample_frames(p, 12).astype(np.float32)
        r = {"clip": p.name, "base": base, "cond": cond, "set": "phys",
             "M": round(float(np.abs(np.diff(v, axis=0)).mean()), 3)}
        for k, q in VARIANTS.items():
            r[k] = round(j.score(p, q), 3)
        rows.append(r)
        print(f"  {p.stem:<42} " + " ".join(f"{k}={r[k]:+.2f}" for k in VARIANTS))

    # --- the valid restitution sweep, for rho(score, e)
    cl = [c for c in json.loads((G0 / "clips.json").read_text())
          if c.get("e") is not None and c["mu"] == 0.5]
    for c in sorted(cl, key=lambda x: (x["object"], x["e"])):
        p = G0 / c["clip"]
        v = sample_frames(p, 12).astype(np.float32)
        r = {"clip": c["clip"], "base": c["object"], "cond": "sweep", "set": "sweep",
             "e": c["e"], "M": round(float(np.abs(np.diff(v, axis=0)).mean()), 3)}
        for k, q in VARIANTS.items():
            r[k] = round(j.score(p, q), 3)
        rows.append(r)
        print(f"  {c['clip']:<42} e={c['e']:.3f}  " +
              " ".join(f"{k}={r[k]:+.2f}" for k in VARIANTS))

    (OUT / "scores.json").write_text(json.dumps(rows, indent=2))

    sp = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)),
                                        np.argsort(np.argsort(b)))[0, 1])
    ph = [r for r in rows if r["set"] == "phys"]
    print(f"\n{'condition':<13} {'n':>2} " + " ".join(f"{k:>10}" for k in VARIANTS) +
          f" {'motion M':>9}")
    print("-" * 62)
    order = ["permanence", "valid", "teleport", "shape"]
    means = {}
    for c in order:
        rs = [r for r in ph if r["cond"] == c]
        means[c] = {k: float(np.mean([r[k] for r in rs])) for k in VARIANTS}
        means[c]["M"] = float(np.mean([r["M"] for r in rs]))
        print(f"{c:<13} {len(rs):>2} " +
              " ".join(f"{means[c][k]:>+10.3f}" for k in VARIANTS) +
              f" {means[c]['M']:>9.2f}")

    print("\nper-clip change from its OWN valid twin (negative = violation detected):")
    for c in ["permanence", "shape", "teleport"]:
        d = []
        for r in [x for x in ph if x["cond"] == c]:
            b = next((x for x in ph if x["base"] == r["base"] and x["cond"] == "valid"),
                     None)
            if b:
                d.append({k: r[k] - b[k] for k in VARIANTS})
        print(f"  {c:<12} " +
              " ".join(f"{k} {np.mean([x[k] for x in d]):+.3f}" for k in VARIANTS))

    print("\nis the score still just motion magnitude?")
    for k in VARIANTS:
        P = [means[c][k] for c in order]
        M = [means[c]["M"] for c in order]
        print(f"  {k:<11} rho(score, M) across conditions = {sp(P, M):+.3f}"
              f"   (Nano: +1.000 for plausible)")
    sw = [r for r in rows if r["set"] == "sweep"]
    for obj in sorted({r["base"] for r in sw}):
        rs = sorted([r for r in sw if r["base"] == obj], key=lambda x: x["e"])
        for k in VARIANTS:
            print(f"  {obj:<13} {k:<11} rho(s,e)={sp([r[k] for r in rs], [r['e'] for r in rs]):+.3f}"
                  f"  rho(s,M)={sp([r[k] for r in rs], [r['M'] for r in rs]):+.3f}")
    print(f"\nwrote {OUT}/scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
