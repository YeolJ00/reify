"""Score authored physics violations with the Cosmos 3 cookbook plausibility question.

Every clip in the sweeps is a valid Newton rollout, so "12/12 possible" proves nothing on
its own -- a model that always answers (A) scores the same. These controls author specific,
unambiguous violations of the three principles the prompt names, through the same Cycles
pipeline and the same scene, so the only difference from the valid clip is the violation:

    permanence  the object is dropped 50 m below the floor for 4 frames, so it vanishes
                and reappears
    shape       the object swells to 2.2x over the second half and stays there
    teleport    the object jumps 0.45 m sideways between two frames

Unlike time reversal -- which the human correctly rejected, since a single bounce played
backwards is a perfectly plausible single bounce -- there is no reading under which an
object doubling in size mid-flight is possible. That is what makes these usable as ground
truth.

Reported alongside is motion magnitude, because the plausibility score on VALID clips
tracks it at rho = -0.94 and any apparent physics signal has to be shown not to be that.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/phys_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "judge" / "phys"
COND = ["valid", "permanence", "shape", "teleport"]


def encode():
    import imageio.v2 as imageio
    made = []
    for d in sorted(LAB.glob("sim_*")):
        if not d.is_dir():
            continue
        ps = sorted(d.glob("f*.png"))
        if not ps:
            continue
        dst = LAB / f"{d.name[4:]}.mp4"
        w = imageio.get_writer(str(dst), fps=24, codec="libx264", quality=8,
                               macro_block_size=1)
        for p in ps:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        made.append(dst.name)
    return made


def main():
    from src.judge.plausibility import PlausibilityJudge, QUESTION
    from src.render.motion_budget import sample_frames

    made = encode()
    print(f"encoded {len(made)} clips")

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
    j = PlausibilityJudge()
    rows = []
    for p in sorted(LAB.glob("*.mp4")):
        base, cond = p.stem.rsplit("__", 1)
        v = sample_frames(p, 12).astype(np.float32)
        M = float(np.abs(np.diff(v, axis=0)).mean())
        r = {"clip": p.name, "base": base, "cond": cond, "M": round(M, 3)}
        for name, q in VARIANTS.items():
            r[name] = round(j.score(p, q), 3)
        rows.append(r)
        print(f"  {p.stem:<40} M={M:>6.2f}  " +
              "  ".join(f"{k}={r[k]:+.2f}" for k in VARIANTS))
    (LAB / "scores.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'condition':<13} {'n':>2} " +
          " ".join(f"{k:>10}" for k in VARIANTS) + f" {'motion M':>9}")
    print("-" * 62)
    for c in COND:
        rs = [r for r in rows if r["cond"] == c]
        if not rs:
            continue
        print(f"{c:<13} {len(rs):>2} " +
              " ".join(f"{np.mean([r[k] for r in rs]):>+10.3f}" for k in VARIANTS) +
              f" {np.mean([r['M'] for r in rs]):>9.2f}")

    print("\nper-clip drop from its own valid twin (negative = violation detected):")
    for c in COND[1:]:
        d = []
        for r in [x for x in rows if x["cond"] == c]:
            base = next((x for x in rows if x["base"] == r["base"]
                         and x["cond"] == "valid"), None)
            if base:
                d.append({k: r[k] - base[k] for k in VARIANTS} |
                         {"dM": r["M"] - base["M"]})
        if d:
            print(f"  {c:<12} " +
                  " ".join(f"{k} {np.mean([x[k] for x in d]):+.3f}" for k in VARIANTS) +
                  f"   dM {np.mean([x['dM'] for x in d]):+.2f}")
    print(f"\nwrote {LAB}/scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
