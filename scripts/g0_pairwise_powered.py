"""G0 rerun, properly powered, with the criterion PRE-REGISTERED (committed before running).

Why a rerun. G0 v1 failed on criterion 4: |rho(margin, dM)| > |rho(margin, de)| for the
brass pot at mu=0.9. That criterion is a bad instrument -- comparing raw marginal
correlations of two CORRELATED predictors does not test whether an effect is really
magnitude. The partial correlation does. On that same cell it gave -0.804, strong and
correctly signed. The failing cell also had n=5 matched pairs, on a grid where Spearman
can only take 21 values.

The criterion below was written and committed BEFORE this script was run. Rewriting a test
after seeing it fail is how the J2 pass survived as long as it did, so the pre-registration
is in version control rather than in a comment written afterwards.

Rule 3 note: these are the 200 EEVEE clips cached by the J3 CEM. EEVEE is a different
render style from the Cycles clips G0 v1 used, so this is its own calibration and does not
inherit or extend that one.

===============================  PRE-REGISTERED  ===============================
DATA
  200 cached CEM clips: 100 rubber duck, 100 brass pot. e in [0.00, 0.64],
  mu in [0.10, 1.51], cd in [343, 15463]. Each object split at ITS OWN median mu
  into a LOW and a HIGH friction half (50 clips each).

PAIRS
  Within one object and one half only. Motion magnitude M = mean absolute pixel
  difference over the 4 fps frames the judge actually receives. Magnitude-matched
  subset = pairs whose |d log M| is at or below the 30th percentile of that cell.
  From that subset, N_PAIRS = 80 sampled uniformly at random with seed 0.
  Every pair scored in BOTH orders and over 2 prompts (4 calls each).

PRIMARY STATISTIC
  Partial Spearman  rho( margin , de | d log M , d mu )
  i.e. rank-transform, residualise margin and de on BOTH controls by least squares,
  correlate the residuals. Controlling for d mu as well as d log M is deliberate:
  friction is the nuisance parameter that inverted the absolute-score result in J3.
  95% CI by bootstrap over pairs, 2000 resamples, seed 0.

PASS  iff in BOTH friction halves, for BOTH objects:
  (a) partial > 0 for rubber_duck, with its 95% CI excluding 0
  (b) partial < 0 for brass_pot,   with its 95% CI excluding 0
  (c) gap = partial_duck - partial_pot > 0.5

FAIL otherwise. Per CLAUDE.md G0, FAIL ⇒ gradient MAP through this judge is dead on
calibration grounds; stop and report, do not build G1.

Everything else -- raw rho with de, raw rho with d log M, position bias, prompt spread,
ceramic-vase behaviour -- is REPORTED for transparency and does NOT enter the verdict.
================================================================================

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python \
       scripts/g0_pairwise_powered.py
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

SRC = REPO / "outputs" / "judge" / "j3"
OUT = REPO / "outputs" / "judge" / "g0b"
OBJS = ("rubber_duck", "brass_pot")
N_PAIRS = 80
MATCH_PCT = 30.0
SEED = 0
N_BOOT = 2000


def motion_magnitude(path, fps=4.0):
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(path))
    src = float(rd.get_meta_data().get("fps", 30.0))
    step = max(int(round(src / fps)), 1)
    fr = [f[..., :3].astype(np.float32) for i, f in enumerate(rd) if i % step == 0]
    rd.close()
    return float(np.abs(np.diff(np.stack(fr), axis=0)).mean())


def _rank(a):
    a = np.asarray(a, float)
    return np.argsort(np.argsort(a)).astype(float)


def partial_spearman(y, x, controls):
    """Spearman of y vs x with the controls removed from both, by rank residualisation."""
    Y, X = _rank(y), _rank(x)
    C = np.column_stack([_rank(c) for c in controls] + [np.ones(len(Y))])
    ry = Y - C @ np.linalg.lstsq(C, Y, rcond=None)[0]
    rx = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
    if np.std(ry) < 1e-9 or np.std(rx) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def boot_ci(y, x, c1, c2, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, x, c1, c2 = map(np.asarray, (y, x, c1, c2))
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = partial_spearman(y[i], x[i], [c1[i], c2[i]])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n // 4:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_cells():
    res = json.loads((SRC / "results.json").read_text())
    recs = [r for _o, rs in res.items() for x in rs for h in x["history"]
            for r in h["records"]]
    have = {p.stem for p in SRC.glob("*.mp4")}
    cells = {}
    for obj in OBJS:
        rs = [r for r in recs if r["valid"] and r["key"] in have
              and r["key"].startswith(obj)]
        for r in rs:
            r["path"] = SRC / f"{r['key']}.mp4"
            r["M"] = motion_magnitude(r["path"])
        med = float(np.median([r["mu"] for r in rs]))
        cells[(obj, "low_mu")] = [r for r in rs if r["mu"] < med]
        cells[(obj, "high_mu")] = [r for r in rs if r["mu"] >= med]
    return cells


def build_pairs(cells):
    rng = np.random.default_rng(SEED)
    work = []
    for (obj, half), cs in cells.items():
        allp = []
        for x, y in itertools.combinations(cs, 2):
            dM = np.log(max(x["M"], 1e-9)) - np.log(max(y["M"], 1e-9))
            allp.append({"object": obj, "half": half, "x": x, "y": y,
                         "de": x["e"] - y["e"], "dM": dM,
                         "dmu": np.log(x["mu"]) - np.log(y["mu"])})
        thr = np.percentile([abs(p["dM"]) for p in allp], MATCH_PCT)
        matched = [p for p in allp if abs(p["dM"]) <= thr]
        idx = rng.permutation(len(matched))[:N_PAIRS]
        work.extend(matched[i] for i in idx)
    return work


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(__doc__.split("=" * 31)[1])          # echo the pre-registration
    cells = build_cells()
    for k, v in sorted(cells.items()):
        e = [r["e"] for r in v]
        mu = [r["mu"] for r in v]
        print(f"  {k[0]:<13} {k[1]:<8} n={len(v):>3}  e {min(e):.3f}..{max(e):.3f}  "
              f"mu {min(mu):.2f}..{max(mu):.2f}")
    work = build_pairs(cells)
    print(f"\n{len(work)} pairs, {4*len(work)} judge calls")

    from src.judge.pairwise import PairwiseJudge
    t0 = time.time()
    j = PairwiseJudge()
    print(f"judge loaded in {time.time()-t0:.0f}s\n")

    t0 = time.time()
    for i, w in enumerate(work):
        r = j.compare(w["x"]["path"], w["y"]["path"], OBJECTS[w["object"]]["noun"])
        w.update(margin=r["margin"], bias=r["bias"], spread=r["spread"])
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(work)}  ({(time.time()-t0)/(i+1):.2f} s/pair)")
    dt = time.time() - t0
    print(f"\n{len(work)} pairs in {dt/60:.1f} min ({dt/len(work):.2f} s/pair)")
    print("decode check:", j._checked)

    (OUT / "pairs.json").write_text(json.dumps(
        [{**{k: v for k, v in w.items() if k not in ("x", "y")},
          "x": w["x"]["key"], "y": w["y"]["key"]} for w in work],
        indent=2, default=float))

    print(f"\n{'half':<9} {'object':<13} {'n':>3} {'partial':>8} {'95% CI':>18} "
          f"{'rho(m,de)':>10} {'rho(m,dM)':>10} {'|bias|':>7}")
    print("-" * 84)
    stat = {}
    for half in ("low_mu", "high_mu"):
        for obj in OBJS:
            g = [w for w in work if w["object"] == obj and w["half"] == half]
            m = [w["margin"] for w in g]
            de = [w["de"] for w in g]
            dM = [w["dM"] for w in g]
            dmu = [w["dmu"] for w in g]
            p = partial_spearman(m, de, [dM, dmu])
            lo, hi = boot_ci(m, de, dM, dmu)
            stat[(half, obj)] = {"n": len(g), "partial": p, "ci": [lo, hi],
                                 "rho_e": spearman(m, de), "rho_M": spearman(m, dM),
                                 "bias": float(np.mean([abs(w["bias"]) for w in g]))}
            s = stat[(half, obj)]
            print(f"{half:<9} {obj:<13} {len(g):>3} {p:>+8.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  {s['rho_e']:>+10.3f} {s['rho_M']:>+10.3f} "
                  f"{s['bias']:>7.2f}")
        print()

    print("=" * 84)
    halves_ok, detail = [], []
    for half in ("low_mu", "high_mu"):
        d, p = stat[(half, "rubber_duck")], stat[(half, "brass_pot")]
        a = d["partial"] > 0 and d["ci"][0] > 0
        b = p["partial"] < 0 and p["ci"][1] < 0
        gap = d["partial"] - p["partial"]
        ok = bool(a and b and gap > 0.5)
        halves_ok.append(ok)
        detail.append({"half": half, "duck": d["partial"], "duck_ci": d["ci"],
                       "pot": p["partial"], "pot_ci": p["ci"], "gap": gap, "ok": ok})
        print(f"{half}: duck {d['partial']:+.3f} (CI excludes 0: {'YES' if a else 'NO'}) | "
              f"pot {p['partial']:+.3f} (CI excludes 0: {'YES' if b else 'NO'}) | "
              f"gap {gap:+.3f} | {'PASS' if ok else 'FAIL'}")
    verdict = "PASS" if all(halves_ok) else "FAIL"
    print(f"\nG0 VERDICT: {verdict}   ({sum(halves_ok)}/2 friction halves)")
    if verdict == "FAIL":
        print("  Per CLAUDE.md G0: stop and report, do not build G1.")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "halves": detail,
         "params": {"n_pairs": N_PAIRS, "match_pct": MATCH_PCT, "seed": SEED,
                    "n_boot": N_BOOT}}, indent=2, default=float))
    print(f"\nwrote {OUT}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
