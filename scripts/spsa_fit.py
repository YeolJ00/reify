"""SPSA fit of theta against the Cosmos3-Super plausibility score.

Run at the human's direction after the calibration failed. That is worth stating plainly
rather than hiding: the band test found the judge's implied prior over restitution has a
proper interior mode (e ~ 0.15..0.24) but that the mode does NOT move with the material
(duck 0.151 vs pot 0.155 at low friction; 0.166 vs 0.238 at high, the wrong way round).

So this run has a falsifiable prediction attached, recorded before it starts:

    IF the prior is material-independent, SPSA started from the same default will
    converge to roughly the SAME theta for the rubber duck and the brass pot, landing
    near e ~ 0.15..0.24 for both.

    IF it converges to materially different theta -- duck clearly bouncier than pot --
    then the calibration statistic was too strict and the conclusion was wrong.

Either way the run is informative, which is why it is worth doing.

Why SPSA rather than gradients or CEM. One evaluation costs a Newton rollout, a Cycles
render and a 33B judge forward, so the evaluation count IS the budget:
    CEM                 population ~ O(d) per step
    finite differences  2d per step
    SPSA                2 per step, independent of d
Backprop is unavailable regardless: the judge runs int8, and a backward pass through a 33B
model at ~1500 video tokens does not fit the card.

theta = (log10 contact damping, log10 friction). Bounds keep the search inside the region
the calibration actually covered; extrapolating outside it would be scoring clips the
judge was never characterised on.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/spsa_fit.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.optimize.spsa import SPSA  # noqa: E402

WARP_PY = "/home/jooyeolyun/anaconda3/envs/warp/bin/python"
BLENDER = "/home/nas5/jaeseonglee/blender-4.2.0-linux-x64/blender"
RUN = REPO / "outputs" / "judge" / "spsa"
MODEL = "nvidia/Cosmos3-Super"
PROMPT = ("Is the way this object moves consistent with what it is made of? Consider its "
          "weight, hardness and how much such a material should bounce. Assume the normal "
          "laws of physics.\nYour answer should be based on the events in the video and "
          "ignore the quality of the simulation engine.\n(A) Consistent\n(B) Inconsistent")

# theta = (log10 cd, log10 mu). DEFAULT START, deliberately the same for both objects so
# any divergence between them comes from the judge and not from initialisation.
X0 = np.array([np.log10(2500.0), np.log10(0.5)])
BOUNDS = np.array([[3.00, 4.10],     # cd 1000 .. 12600
                   [-0.80, 0.05]])   # mu 0.16 .. 1.12
N_ITER = 14
A_GAIN, C_GAIN = 0.05, 0.15


def to_theta(x):
    return {"cd": float(10.0 ** x[0]), "mu": float(10.0 ** x[1])}


def run_batch(judge, obj, xs, tag):
    """Simulate, render and score a list of theta. Returns list of dicts (None if unusable)."""
    from src.render.motion_budget import sample_frames, check

    batch = {f"{tag}_{i}": {"object": obj, **to_theta(x)} for i, x in enumerate(xs)}
    (RUN / "batch_in.json").write_text(json.dumps(batch))
    r = subprocess.run([WARP_PY, str(REPO / "scripts" / "spsa_sim_batch.py"), str(RUN)],
                       cwd=REPO, capture_output=True, text=True, env=dict(os.environ))
    if r.returncode != 0:
        raise RuntimeError(f"sim failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    meta = json.loads((RUN / "batch_out.json").read_text())

    for d in RUN.glob("sim_*"):
        if d.is_dir():
            shutil.rmtree(d)
    for p in RUN.glob("*.mp4"):
        p.unlink()
    rr = subprocess.run([BLENDER, "--background", "--python",
                         str(REPO / "scripts" / "blender_render_sim.py")],
                        cwd=REPO, capture_output=True, text=True,
                        env={**os.environ, "LAB": str(RUN)})
    if rr.returncode != 0:
        raise RuntimeError(f"render failed:\n{rr.stdout[-1500:]}")

    import imageio.v2 as imageio
    out = []
    for i, k in enumerate(batch):
        d = RUN / f"sim_{k}"
        ps = sorted(d.glob("f*.png")) if d.is_dir() else []
        if not meta.get(k, {}).get("ok") or not ps:
            out.append(None)
            continue
        dst = RUN / f"{k}.mp4"
        w = imageio.get_writer(str(dst), fps=24, codec="libx264", quality=8,
                               macro_block_size=1)
        for p in ps:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        ok, frac, _ = check(dst)
        v = sample_frames(dst, 12).astype(np.float32)
        rec = {"key": k, "x": [float(z) for z in xs[i]], **to_theta(xs[i]),
               **meta[k], "guard_ok": bool(ok), "motion_fraction": round(frac, 3),
               "M": round(float(np.abs(np.diff(v, axis=0)).mean()), 3)}
        rec["s"] = float(judge.score(dst, PROMPT))
        out.append(rec)
        shutil.rmtree(d)
    return out


def main():
    from src.judge.plausibility import PlausibilityJudge

    RUN.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "g0c" / "lab.json", RUN / "lab.json")
    print(__doc__.split("Run:")[0])
    t0 = time.time()
    judge = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"judge loaded in {time.time()-t0:.0f}s\n")

    results = {}
    for obj in ("rubber_duck", "brass_pot"):
        print(f"=== {obj}   start cd={10**X0[0]:.0f} mu={10**X0[1]:.2f}")
        trace = []
        state = {"k": 0}

        def objective(x, _obj=obj, _tr=trace, _st=state):
            recs = run_batch(judge, _obj, [x], f"{_obj}_e{len(_tr):03d}")
            r = recs[0]
            if r is None:
                # unsimulatable theta: strongly discouraged, not silently skipped
                _tr.append({"x": [float(z) for z in x], "s": None, "valid": False})
                return -10.0
            _tr.append(r)
            # rule 4: motion magnitude logged on every evaluation, never trusted blindly
            print(f"    eval {len(_tr):>3}  cd={r['cd']:>7.0f} mu={r['mu']:.2f} "
                  f"e={r['e']:.3f} bounces={r['bounces']:>2} M={r['M']:.2f}  s={r['s']:+.3f}")
            return r["s"]

        opt = SPSA(objective, X0.copy(), BOUNDS, a=A_GAIN, c=C_GAIN, n_iter=N_ITER,
                   seed=0, maximize=True,
                   log=lambda m: print("  " + m))
        x, hist = opt.run()
        # report the best EVALUATED point as well as the final iterate
        good = [r for r in trace if r.get("s") is not None]
        best = max(good, key=lambda r: r["s"]) if good else None
        results[obj] = {"x_final": [float(z) for z in x],
                        "theta_final": to_theta(x),
                        "best": best, "trace": trace,
                        "n_eval": len(trace)}
        print(f"  -> final cd={10**x[0]:.0f} mu={10**x[1]:.2f}"
              + (f" | best evaluated cd={best['cd']:.0f} mu={best['mu']:.2f} "
                 f"e={best['e']:.3f} s={best['s']:+.3f}" if best else ""))
        (RUN / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== PREDICTION CHECK")
    print("  calibration said the prior does not condition on material, so both objects")
    print("  should land near the same theta (e ~ 0.15..0.24).")
    for obj, r in results.items():
        b = r["best"]
        print(f"  {obj:<13} final cd={r['theta_final']['cd']:>7.0f} "
              f"mu={r['theta_final']['mu']:.2f}"
              + (f"   best e={b['e']:.3f} bounces={b['bounces']}" if b else ""))
    if len(results) == 2:
        a, b = results["rubber_duck"], results["brass_pot"]
        if a["best"] and b["best"]:
            de = a["best"]["e"] - b["best"]["e"]
            print(f"\n  e(duck) - e(pot) at the best evaluated point = {de:+.3f}")
            print("  positive and sizeable => calibration was too strict, I was wrong")
            print("  near zero or negative => material-independent prior, as measured")
    (RUN / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RUN}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
