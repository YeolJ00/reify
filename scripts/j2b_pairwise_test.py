"""G0 / J2b: magnitude-matched PAIRWISE kill test. Gates the whole G-track.

Pre-registered before running. Absolute-score calibration (J2) failed twice over, and the
second failure is the one this has to answer: the judge's restitution preference for the
rubber duck INVERTED (rho = -1.000) when friction alone moved from mu 0.9 to mu ~0.4, same
engine, same judge, same prompts, same damping values. A calibration measured at one slice
of a nuisance parameter told us nothing about the next slice.

So this runs at BOTH cached friction slices and requires the result to hold at both.

Design
------
* Pairs are within one object and one friction slice, so object identity and appearance are
  constant inside every comparison and cancel in the margin.
* MAGNITUDE MATCHING. The known hacking ghost is that the judge picks the clip that moves
  more. Motion magnitude M is measured where the judge actually sees it -- mean absolute
  pixel difference between consecutive frames of the 4 fps decode it receives, not a
  simulator-side quantity. The primary readout uses only pairs whose |log M| gap is small,
  where a magnitude-reader must score at chance.
* Both orders, so position bias is removed rather than averaged over.

Readouts
--------
1. rho(margin, delta_e) per object per slice -- does preferring a clip track restitution?
2. SIGN FLIP across objects: positive for the rubber duck, negative for the brass pot.
3. ROBUSTNESS: (2) must hold at BOTH friction slices.
4. CONFOUND: on the matched subset, |rho(margin, delta_M)| < |rho(margin, delta_e)|.

PASS requires 2, 3 and 4. FAIL ⇒ gradient MAP through this judge is dead on calibration
grounds; stop and report (CLAUDE.md G0).

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python \
       scripts/j2b_pairwise_test.py [--match-frac 0.5]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import OBJECTS  # noqa: E402

OUT = REPO / "outputs" / "judge" / "g0"

# Two cached slices of the nuisance parameter, both rendered in Cycles.
SLICES = {
    "mu=0.9": {"dir": REPO / "outputs" / "judge" / "j2r",
               "cells": REPO / "outputs" / "judge" / "j2r" / "scored.json",
               "objects": ["rubber_duck", "baseball", "ceramic_vase", "brass_pot"]},
    "mu~0.4": {"dir": REPO / "outputs" / "judge" / "j3conf",
               "cells": REPO / "outputs" / "judge" / "j3conf" / "confirm.json",
               "objects": ["rubber_duck", "brass_pot"]},
}


def load_cells(spec):
    """Both caches carry e per clip; only their shapes differ."""
    raw = json.loads(Path(spec["cells"]).read_text())
    rows = []
    if isinstance(raw, list):                      # j2r/scored.json
        for r in raw:
            rows.append({"key": r["key"], "object": r["object"], "e": r["e"]})
    else:                                          # j3conf/confirm.json
        for k, v in raw.items():
            rows.append({"key": k, "object": v["object"], "e": v["e"]})
    return [r for r in rows if r["object"] in spec["objects"] and r["e"] is not None]


def motion_magnitude(path, fps=4.0):
    """Mean |frame difference| over the decode the judge receives. This is the quantity a
    magnitude-reading judge would be responding to, measured in pixels rather than inferred
    from theta."""
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(path))
    src = float(rd.get_meta_data().get("fps", 30.0))
    step = max(int(round(src / fps)), 1)
    fr = [f[..., :3].astype(np.float32) for i, f in enumerate(rd) if i % step == 0]
    rd.close()
    v = np.stack(fr)
    return float(np.abs(np.diff(v, axis=0)).mean())


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-frac", type=float, default=0.5,
                    help="fraction of pairs kept as magnitude-matched (smallest |dlogM|)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- build the pair list first, so the cost is known before the model loads
    work = []
    for sl, spec in SLICES.items():
        cells = load_cells(spec)
        for c in cells:
            c["path"] = spec["dir"] / f"{c['key']}.mp4"
            c["M"] = motion_magnitude(c["path"])
        for obj in spec["objects"]:
            cs = [c for c in cells if c["object"] == obj]
            for x, y in itertools.combinations(cs, 2):
                work.append({"slice": sl, "object": obj, "x": x, "y": y,
                             "de": x["e"] - y["e"],
                             "dM": np.log(max(x["M"], 1e-6)) - np.log(max(y["M"], 1e-6))})
    print(f"{len(work)} pairs, {4*len(work)} judge calls "
          f"(2 prompts x 2 orders)\n")
    for sl in SLICES:
        n = sum(1 for w in work if w["slice"] == sl)
        print(f"  {sl}: {n} pairs")

    from src.judge.pairwise import PairwiseJudge
    t0 = time.time()
    j = PairwiseJudge()
    print(f"\njudge loaded in {time.time()-t0:.0f}s")

    t0 = time.time()
    for i, w in enumerate(work):
        r = j.compare(w["x"]["path"], w["y"]["path"], OBJECTS[w["object"]]["noun"])
        w.update(margin=r["margin"], bias=r["bias"], spread=r["spread"])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(work)} pairs  ({(time.time()-t0)/(i+1):.2f} s/pair)")
    dt = time.time() - t0
    print(f"\n{len(work)} pairs in {dt:.0f}s ({dt/len(work):.2f} s/pair)")
    print("decode check:", j._checked)

    rec = [{k: (v if k not in ("x", "y") else
                {"key": v["key"], "e": v["e"], "M": v["M"]}) for k, v in w.items()}
           for w in work]
    (OUT / "pairs.json").write_text(json.dumps(rec, indent=2, default=float))

    # ---- readouts
    print(f"\n{'slice':<9} {'object':<14} {'n':>3} {'set':<9} "
          f"{'rho(m,de)':>10} {'rho(m,dM)':>10} {'|bias|':>7}")
    print("-" * 74)
    res = {}
    for sl in SLICES:
        for obj in SLICES[sl]["objects"]:
            ws = [w for w in work if w["slice"] == sl and w["object"] == obj]
            if len(ws) < 3:
                continue
            k = int(max(3, round(a.match_frac * len(ws))))
            matched = sorted(ws, key=lambda w: abs(w["dM"]))[:k]
            for tag, group in (("all", ws), ("matched", matched)):
                m = [w["margin"] for w in group]
                r_e = spearman(m, [w["de"] for w in group])
                r_M = spearman(m, [w["dM"] for w in group])
                bias = float(np.mean([abs(w["bias"]) for w in group]))
                print(f"{sl:<9} {obj:<14} {len(group):>3} {tag:<9} "
                      f"{r_e:>+10.3f} {r_M:>+10.3f} {bias:>7.2f}")
                res[(sl, obj, tag)] = {"n": len(group), "rho_e": r_e, "rho_M": r_M,
                                       "bias": bias}
            print()

    # ---- verdict
    print("=" * 74)
    ok_slices, detail = [], []
    for sl in SLICES:
        if not {"rubber_duck", "brass_pot"} <= set(SLICES[sl]["objects"]):
            continue
        d = res.get((sl, "rubber_duck", "matched"))
        p = res.get((sl, "brass_pot", "matched"))
        if d is None or p is None:
            continue
        flip = d["rho_e"] > 0 > p["rho_e"]
        gap = d["rho_e"] - p["rho_e"]
        conf_ok = all(abs(x["rho_M"]) < abs(x["rho_e"]) for x in (d, p))
        ok_slices.append(bool(flip and gap > 0.5 and conf_ok))
        detail.append((sl, d["rho_e"], p["rho_e"], gap, flip, conf_ok))
        print(f"{sl}: duck {d['rho_e']:+.3f} | pot {p['rho_e']:+.3f} | gap {gap:+.3f} "
              f"| sign flip {'YES' if flip else 'NO'} | not magnitude {'YES' if conf_ok else 'NO'}")
    verdict = "PASS" if (len(ok_slices) >= 2 and all(ok_slices)) else "FAIL"
    print(f"\nrobust across {sum(ok_slices)}/{len(ok_slices)} friction slices")
    print(f"G0 VERDICT: {verdict}")
    if verdict == "FAIL":
        print("  Per CLAUDE.md G0: gradient MAP through this judge is dead on "
              "calibration grounds. Stop and report.")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "slices": [
            {"slice": s, "rho_duck": rd, "rho_pot": rp, "gap": g,
             "sign_flip": f, "not_magnitude": c} for s, rd, rp, g, f, c in detail],
         "match_frac": a.match_frac}, indent=2, default=float))
    print(f"\nwrote {OUT}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
