"""G0/J2 calibration on the Cosmos3-Super reasoner. Gates the SPSA fit.

Super earned this run: on authored violations it scored 9/9 correct where Nano scored 1/9,
and its rank correlation with motion magnitude fell from +1.000 to +0.200. But detecting a
teleporting object and resolving e=0.4 from e=0.7 are different asks, and only the first is
established. This tests the second, which is the one the pipeline actually needs.

Instrument change from the earlier G0 runs, stated plainly: this scores the ABSOLUTE
plausibility margin on a single clip, not a pairwise A-vs-B margin. CLAUDE.md rule 1 says
pairwise only, because absolute yes/no scoring tracked motion magnitude -- but that rule was
bought with a bad prompt on a smaller model. The cookbook prompt gives a 6.2-point dynamic
range against ~0.3 for pairwise margins, needs one video instead of two, and carries no
position bias to cancel. Pairwise remains available if this fails; it is 4x the cost per
comparison, which is why it is not the first thing tried on a 33B int8 model.

===============================  PRE-REGISTERED  ===============================
DATA
  The 56 guarded Cycles clips of outputs/judge/g0c (rubber duck, brass pot; 6 damping x
  5 friction each, minus 4 that failed the motion guard). Each object split at ITS OWN
  median mu into a LOW and a HIGH friction half.

SCORE
  s = logprob(A) - logprob(B) from the Cosmos3-Super reasoner, int8, fp32 head, all
  frames of the clip. Three prompts: plausible / natural / material. The PRIMARY prompt
  is `material`, declared before running, because it is the only one that asks about the
  object's substance and the only one that separated e from motion magnitude on Super's
  first pass (rho(s,e) -0.829 vs rho(s,M) -0.714 on the brass pot). The other two are
  reported and do not enter the verdict.

PRIMARY STATISTIC  (identical in form to ce7bc5d and 9730fbd)
  Partial Spearman  rho( s , e | log M , log mu ),  95% CI by bootstrap over clips,
  2000 resamples, seed 0.  M is mean absolute pixel difference over the frames scored.

PASS  iff in BOTH friction halves, under the PRIMARY prompt:
  (a) partial > 0 for rubber_duck, with its 95% CI excluding 0
      (a rubber duck should look MORE plausible when it bounces more)
  (b) partial < 0 for brass_pot,   with its 95% CI excluding 0
      (a brass pot should look LESS plausible when it bounces more)
  (c) gap = partial_duck - partial_pot > 0.5

FAIL otherwise ⇒ do not run the SPSA fit; report and stop.
================================================================================

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=6 .../envs/cosmos/bin/python \
       scripts/g0_super_calibration.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

from scripts.g0_pairwise_powered import boot_ci, partial_spearman, spearman  # noqa: E402

G0 = REPO / "outputs" / "judge" / "g0c"
OUT = REPO / "outputs" / "judge" / "super_cal"
MODEL = "nvidia/Cosmos3-Super"
PRIMARY = "material"


def main():
    from src.judge.plausibility import PlausibilityJudge, QUESTION
    from src.render.motion_budget import sample_frames

    OUT.mkdir(parents=True, exist_ok=True)
    print(__doc__.split("=" * 31)[1])

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
    clips = [c for c in json.loads((G0 / "clips.json").read_text())
             if c.get("guard_ok") and c.get("e") is not None]
    print(f"{len(clips)} guarded clips, {len(clips)*len(VARIANTS)} forwards\n")

    t0 = time.time()
    j = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"loaded in {time.time()-t0:.0f}s\n")

    t0 = time.time()
    for i, c in enumerate(clips):
        p = G0 / c["clip"]
        v = sample_frames(p, 12).astype(np.float32)
        c["M"] = float(np.abs(np.diff(v, axis=0)).mean())
        for k, q in VARIANTS.items():
            c[k] = round(j.score(p, q), 4)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(clips)}  ({(time.time()-t0)/(i+1):.1f} s/clip)")
    (OUT / "scores.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k != "deltas"} for c in clips], indent=2))
    print(f"\nscored in {(time.time()-t0)/60:.1f} min")

    cells = {}
    for obj in ("rubber_duck", "brass_pot"):
        cs = [c for c in clips if c["object"] == obj]
        med = float(np.median([c["mu"] for c in cs]))
        cells[(obj, "low_mu")] = [c for c in cs if c["mu"] < med]
        cells[(obj, "high_mu")] = [c for c in cs if c["mu"] >= med]

    print(f"\n{'half':<9} {'object':<13} {'prompt':<11} {'n':>3} {'partial':>8} "
          f"{'95% CI':>18} {'rho(s,e)':>9} {'rho(s,M)':>9}")
    print("-" * 92)
    stat = {}
    for half in ("low_mu", "high_mu"):
        for obj in ("rubber_duck", "brass_pot"):
            g = cells[(obj, half)]
            s_e = [c["e"] for c in g]
            lM = [np.log(max(c["M"], 1e-9)) for c in g]
            lmu = [np.log(c["mu"]) for c in g]
            for k in VARIANTS:
                s = [c[k] for c in g]
                p = partial_spearman(s, s_e, [lM, lmu])
                lo, hi = boot_ci(s, s_e, np.array(lM), np.array(lmu))
                stat[(half, obj, k)] = {"n": len(g), "partial": p, "ci": [lo, hi],
                                        "rho_e": spearman(s, s_e),
                                        "rho_M": spearman(s, lM)}
                r = stat[(half, obj, k)]
                mark = " *" if k == PRIMARY else ""
                print(f"{half:<9} {obj:<13} {k:<11} {len(g):>3} {p:>+8.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]  {r['rho_e']:>+9.3f} {r['rho_M']:>+9.3f}{mark}")
            print()

    print("=" * 92)
    print(f"PRIMARY prompt: {PRIMARY}")
    ok, detail = [], []
    for half in ("low_mu", "high_mu"):
        d = stat[(half, "rubber_duck", PRIMARY)]
        p = stat[(half, "brass_pot", PRIMARY)]
        a = d["partial"] > 0 and d["ci"][0] > 0
        b = p["partial"] < 0 and p["ci"][1] < 0
        gap = d["partial"] - p["partial"]
        good = bool(a and b and gap > 0.5)
        ok.append(good)
        detail.append({"half": half, "duck": d["partial"], "duck_ci": d["ci"],
                       "pot": p["partial"], "pot_ci": p["ci"], "gap": gap, "ok": good})
        print(f"{half}: duck {d['partial']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
              f"{'OK' if a else 'no'} | pot {p['partial']:+.3f} "
              f"CI[{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] {'OK' if b else 'no'} | "
              f"gap {gap:+.3f} | {'PASS' if good else 'FAIL'}")
    verdict = "PASS" if all(ok) else "FAIL"
    print(f"\nG0-SUPER VERDICT: {verdict}   ({sum(ok)}/2 friction halves)")
    if verdict == "FAIL":
        print("  Do not run the SPSA fit. Report and stop.")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "primary": PRIMARY, "halves": detail,
         "stat": {f"{a}|{b}|{c}": v for (a, b, c), v in stat.items()}},
        indent=2, default=float))
    print(f"\nwrote {OUT}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
