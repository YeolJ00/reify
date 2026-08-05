"""G0, third run: pairwise kill test on GUARDED, motion-dense, Cycles clips.

Why a third run. The first two were measured on clips that barely contained the event.
On the clips G0 v2 scored, 1 of 9 frame deltas contained motion for the brass pot and 2 of
9 for the duck -- the drop, bounce and settle all happened between the judge's first and
second sampled frame. That run's conclusion ("the judge has no stable material
preference") was really a statement about clips with no physics in them. Its FAIL is
treated as VOID, not as a verdict.

What changed, all three fixes together:
  * drop height back to 0.30 m, and clips TRIMMED to the event window, so the roll-away
    tail that motivated lowering it is cut instead of the motion being cut;
  * judge decodes a fixed 12 frames spanning the clip rather than a fixed 4 fps, so a
    short trimmed event is fully covered (judge change ⇒ rule 3 ⇒ this recalibration);
  * CYCLES, not EEVEE -- EEVEE Next defaults to use_raytracing=False and renders the
    brass pot as a dark matte lump, a rule-2 violation on the object that flipped hardest.
  * every clip must pass the motion guard (>= 60% of sampled deltas contain motion)
    before it is scored at all.

===============================  PRE-REGISTERED  ===============================
The PRIMARY STATISTIC and the PASS RULE are UNCHANGED from ce7bc5d. Only pair
selection changes, because the clip set changed and the old selection does not fit it.

DATA
  60 Cycles clips (30 rubber duck, 30 brass pot) over a 6 damping x 5 friction grid,
  drop 0.30 m, trimmed to the event window. Clips failing the motion guard are
  DROPPED before any scoring. Each object split at ITS OWN median mu into a LOW and a
  HIGH friction half.

PAIRS
  ALL within-object, within-half pairs (~105 per cell), not a magnitude-matched
  subset. With 15 clips per cell a 30th-percentile matched subset is only ~31 pairs,
  too few for the CI to decide anything; and the primary statistic already controls
  for magnitude by partialling it out, which is what the matching was approximating.
  The magnitude-matched subset is still reported as a secondary check.
  Every pair scored in BOTH orders and over 2 prompts (4 calls each).

PRIMARY STATISTIC  (identical to ce7bc5d)
  Partial Spearman  rho( margin , de | d log M , d mu ),
  95% CI by bootstrap over pairs, 2000 resamples, seed 0.

PASS  iff in BOTH friction halves:
  (a) partial > 0 for rubber_duck, with its 95% CI excluding 0
  (b) partial < 0 for brass_pot,   with its 95% CI excluding 0
  (c) gap = partial_duck - partial_pot > 0.5

FAIL otherwise ⇒ stop, do not build G1.
================================================================================

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/g0c_test.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import OBJECTS  # noqa: E402
from scripts.g0_pairwise_powered import (  # noqa: E402
    boot_ci, partial_spearman, spearman)
from src.render.motion_budget import motion_fraction  # noqa: E402

LAB = REPO / "outputs" / "judge" / "g0c"
OUT = REPO / "outputs" / "judge" / "g0c"
OBJS = ("rubber_duck", "brass_pot")
MATCH_PCT = 30.0


def main():
    print(__doc__.split("=" * 31)[1])
    clips = json.loads((LAB / "clips.json").read_text())
    kept = [c for c in clips if c.get("guard_ok") and c.get("e") is not None]
    print(f"clips {len(clips)}, guard passed {len(kept)}, dropped {len(clips)-len(kept)}")
    for c in kept:
        c["path"] = LAB / c["clip"]
        c["M"] = motion_fraction(c["path"])[0]        # placeholder, replaced below
    # magnitude = mean |delta| over the sampled frames, not the fraction above
    from src.render.motion_budget import sample_frames
    for c in kept:
        v = sample_frames(c["path"]).astype(np.float32)
        c["M"] = float(np.abs(np.diff(v, axis=0)).mean())

    cells = {}
    for obj in OBJS:
        cs = [c for c in kept if c["object"] == obj]
        med = float(np.median([c["mu"] for c in cs]))
        cells[(obj, "low_mu")] = [c for c in cs if c["mu"] < med]
        cells[(obj, "high_mu")] = [c for c in cs if c["mu"] >= med]
    work = []
    for (obj, half), cs in cells.items():
        e = [c["e"] for c in cs]
        print(f"  {obj:<13} {half:<8} n={len(cs):>2}  e {min(e):.3f}..{max(e):.3f}")
        for x, y in itertools.combinations(cs, 2):
            work.append({"object": obj, "half": half, "x": x, "y": y,
                         "de": x["e"] - y["e"],
                         "dM": np.log(max(x["M"], 1e-9)) - np.log(max(y["M"], 1e-9)),
                         "dmu": np.log(x["mu"]) - np.log(y["mu"])})
    print(f"\n{len(work)} pairs, {4*len(work)} judge calls")

    from src.judge.pairwise import PairwiseJudge
    j = PairwiseJudge(n_frames=12)
    t0 = time.time()
    for i, w in enumerate(work):
        r = j.compare(w["x"]["path"], w["y"]["path"], OBJECTS[w["object"]]["noun"])
        w.update(margin=r["margin"], bias=r["bias"], spread=r["spread"])
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(work)}  ({(time.time()-t0)/(i+1):.2f} s/pair)")
    print(f"\n{len(work)} pairs in {(time.time()-t0)/60:.1f} min")
    print("decode check:", j._checked)

    (OUT / "pairs.json").write_text(json.dumps(
        [{**{k: v for k, v in w.items() if k not in ("x", "y")},
          "x": w["x"]["clip"], "y": w["y"]["clip"]} for w in work],
        indent=2, default=float))

    print(f"\n{'half':<9} {'object':<13} {'set':<8} {'n':>4} {'partial':>8} "
          f"{'95% CI':>18} {'rho(m,de)':>10} {'rho(m,dM)':>10} {'|bias|':>7}")
    print("-" * 96)
    stat = {}
    for half in ("low_mu", "high_mu"):
        for obj in OBJS:
            g = [w for w in work if w["object"] == obj and w["half"] == half]
            thr = np.percentile([abs(w["dM"]) for w in g], MATCH_PCT)
            for tag, grp in (("all", g),
                             ("matched", [w for w in g if abs(w["dM"]) <= thr])):
                m = [w["margin"] for w in grp]
                de = [w["de"] for w in grp]
                dM = [w["dM"] for w in grp]
                dmu = [w["dmu"] for w in grp]
                p = partial_spearman(m, de, [dM, dmu])
                lo, hi = boot_ci(m, de, dM, dmu)
                rec = {"n": len(grp), "partial": p, "ci": [lo, hi],
                       "rho_e": spearman(m, de), "rho_M": spearman(m, dM),
                       "bias": float(np.mean([abs(w["bias"]) for w in grp]))}
                stat[(half, obj, tag)] = rec
                print(f"{half:<9} {obj:<13} {tag:<8} {len(grp):>4} {p:>+8.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]  {rec['rho_e']:>+10.3f} "
                      f"{rec['rho_M']:>+10.3f} {rec['bias']:>7.2f}")
        print()

    print("=" * 96)
    ok, detail = [], []
    for half in ("low_mu", "high_mu"):
        d, p = stat[(half, "rubber_duck", "all")], stat[(half, "brass_pot", "all")]
        a = d["partial"] > 0 and d["ci"][0] > 0
        b = p["partial"] < 0 and p["ci"][1] < 0
        gap = d["partial"] - p["partial"]
        good = bool(a and b and gap > 0.5)
        ok.append(good)
        detail.append({"half": half, "duck": d["partial"], "duck_ci": d["ci"],
                       "pot": p["partial"], "pot_ci": p["ci"], "gap": gap,
                       "duck_sig": a, "pot_sig": b, "ok": good})
        print(f"{half}: duck {d['partial']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
              f"{'OK' if a else 'no'} | pot {p['partial']:+.3f} "
              f"CI[{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] {'OK' if b else 'no'} | "
              f"gap {gap:+.3f} | {'PASS' if good else 'FAIL'}")
    verdict = "PASS" if all(ok) else "FAIL"
    print(f"\nG0 VERDICT: {verdict}   ({sum(ok)}/2 friction halves)")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "halves": detail,
         "stat": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in stat.items()}},
        indent=2, default=float))
    print(f"\nwrote {OUT}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
