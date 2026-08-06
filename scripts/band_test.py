"""Peak-location test on the re-centred sweep. Gates the SPSA fit.

Why the statistic changes, and why that is legitimate here rather than goalpost-moving:
the previous test asked whether plausibility rises MONOTONICALLY with restitution. If each
material instead has a plausible restitution and the score PEAKS there, a monotone
statistic is structurally unable to see it -- it was the wrong shape, not a strict enough
threshold. The change is declared here, before this sweep has been scored at all, on data
that does not yet exist. The old verdict stands as a FAIL on its own terms.

===============================  PRE-REGISTERED  ===============================
DATA
  The guarded Cycles clips of outputs/judge/band: rubber duck and brass pot,
  7 damping x 4 friction each, restitution measured per rollout and spanning roughly
  0.03..0.30, where these objects bounce 1-8 times rather than 17-21. Each object split
  at ITS OWN median mu into a LOW and a HIGH friction half.

SCORE
  s = logprob(A) - logprob(B), Cosmos3-Super reasoner, int8, fp32 head, all clip frames.
  PRIMARY prompt `material` (declared before the run, as in the previous registration).

PRIMARY STATISTIC -- peak location, magnitude-controlled
  1. residualise s on log M by least squares, where M is mean absolute pixel difference
     over the scored frames.  s_r = s - (a + b log M)
  2. peak_e = softmax-weighted mean of e,  weights = exp(s_r / T),  T = 1.0
  3. delta = peak_e(rubber_duck) - peak_e(brass_pot)
  95% CI on delta by bootstrap over clips, 2000 resamples, seed 0.

PASS  iff in BOTH friction halves:
  delta > 0  AND its 95% CI excludes 0.
  i.e. the rubber duck's plausibility peak sits at a HIGHER restitution than the brass
  pot's -- a duck should bounce more than a metal pot.

FAIL otherwise ⇒ do not run the SPSA fit; report and stop.

Reported but NOT part of the verdict: raw (non-residualised) peaks, argmax, the monotone
rho(s,e) and rho(s,M), and the other two prompts.
================================================================================

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=6 .../envs/cosmos/bin/python scripts/band_test.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.g0_pairwise_powered import spearman  # noqa: E402

LAB = REPO / "outputs" / "judge" / "band"
OUT = LAB
MODEL = "nvidia/Cosmos3-Super"
PRIMARY = "material"
T_SOFT = 1.0
N_BOOT = 2000


def resid(s, logM):
    s, x = np.asarray(s, float), np.asarray(logM, float)
    A = np.column_stack([x, np.ones(len(x))])
    return s - A @ np.linalg.lstsq(A, s, rcond=None)[0]


def peak_e(e, s, logM, T=T_SOFT):
    r = resid(s, logM)
    w = np.exp((r - r.max()) / T)
    return float(np.sum(w * np.asarray(e, float)) / np.sum(w))


def boot_delta(gd, gp, key, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = rng.integers(0, len(gd["e"]), len(gd["e"]))
        b = rng.integers(0, len(gp["e"]), len(gp["e"]))
        try:
            pd = peak_e(gd["e"][a], gd[key][a], gd["lM"][a])
            pp = peak_e(gp["e"][b], gp[key][b], gp["lM"][b])
            out.append(pd - pp)
        except Exception:
            pass
    if len(out) < n // 4:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    from src.judge.plausibility import PlausibilityJudge, QUESTION
    from src.render.motion_budget import sample_frames

    print(__doc__.split("=" * 31)[1])
    clips = [c for c in json.loads((LAB / "clips.json").read_text())
             if c.get("guard_ok") and c.get("e") is not None]
    print(f"{len(clips)} guarded clips\n")

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
    t0 = time.time()
    j = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"loaded in {time.time()-t0:.0f}s")
    t0 = time.time()
    for i, c in enumerate(clips):
        p = LAB / c["clip"]
        v = sample_frames(p, 12).astype(np.float32)
        c["M"] = float(np.abs(np.diff(v, axis=0)).mean())
        for k, q in VARIANTS.items():
            c[k] = round(j.score(p, q), 4)
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{len(clips)} ({(time.time()-t0)/(i+1):.1f} s/clip)")
    (OUT / "scores.json").write_text(json.dumps(clips, indent=2))
    print(f"scored in {(time.time()-t0)/60:.1f} min\n")

    groups = {}
    for obj in ("rubber_duck", "brass_pot"):
        cs = [c for c in clips if c["object"] == obj]
        med = float(np.median([c["mu"] for c in cs]))
        for half, sel in (("low_mu", [c for c in cs if c["mu"] < med]),
                          ("high_mu", [c for c in cs if c["mu"] >= med])):
            groups[(obj, half)] = {
                "e": np.array([c["e"] for c in sel]),
                "lM": np.log(np.array([max(c["M"], 1e-9) for c in sel])),
                **{k: np.array([c[k] for c in sel]) for k in VARIANTS}}

    print(f"{'half':<9} {'object':<13} {'prompt':<11} {'n':>3} {'peak_e':>8} "
          f"{'raw peak':>9} {'argmax e':>9} {'rho(s,e)':>9} {'rho(s,M)':>9}")
    print("-" * 92)
    peaks = {}
    for half in ("low_mu", "high_mu"):
        for obj in ("rubber_duck", "brass_pot"):
            g = groups[(obj, half)]
            for k in VARIANTS:
                pe = peak_e(g["e"], g[k], g["lM"])
                raw = float(np.sum(np.exp((g[k] - g[k].max()) / T_SOFT) * g["e"]) /
                            np.sum(np.exp((g[k] - g[k].max()) / T_SOFT)))
                peaks[(half, obj, k)] = pe
                mark = " *" if k == PRIMARY else ""
                print(f"{half:<9} {obj:<13} {k:<11} {len(g['e']):>3} {pe:>8.3f} "
                      f"{raw:>9.3f} {g['e'][int(np.argmax(g[k])):][0]:>9.3f} "
                      f"{spearman(g[k], g['e']):>+9.3f} "
                      f"{spearman(g[k], g['lM']):>+9.3f}{mark}")
            print()

    print("=" * 92)
    print(f"PRIMARY prompt: {PRIMARY}   statistic: peak_e(duck) - peak_e(pot), "
          f"magnitude-residualised")
    ok, detail = [], []
    for half in ("low_mu", "high_mu"):
        gd, gp = groups[("rubber_duck", half)], groups[("brass_pot", half)]
        d = peaks[(half, "rubber_duck", PRIMARY)] - peaks[(half, "brass_pot", PRIMARY)]
        lo, hi = boot_delta(gd, gp, PRIMARY)
        good = bool(d > 0 and lo > 0)
        ok.append(good)
        detail.append({"half": half, "peak_duck": peaks[(half, "rubber_duck", PRIMARY)],
                       "peak_pot": peaks[(half, "brass_pot", PRIMARY)],
                       "delta": d, "ci": [lo, hi], "ok": good})
        print(f"{half}: peak duck {peaks[(half,'rubber_duck',PRIMARY)]:.3f}  "
              f"peak pot {peaks[(half,'brass_pot',PRIMARY)]:.3f}  "
              f"delta {d:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]  "
              f"{'PASS' if good else 'FAIL'}")
    verdict = "PASS" if all(ok) else "FAIL"
    print(f"\nBAND VERDICT: {verdict}   ({sum(ok)}/2 friction halves)")
    if verdict != "PASS":
        print("  Do not run the SPSA fit.")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "primary": PRIMARY, "halves": detail}, indent=2,
        default=float))
    print(f"\nwrote {OUT}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
