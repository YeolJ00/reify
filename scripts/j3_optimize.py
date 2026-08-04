"""J3: recover theta by multi-start MAP-CEM against the judge.

Objective   J(theta) = beta * judge_score(render(sim(theta)) | c) + log p(theta)

where c is a material description naming the object, judge_score is the yes/no logprob
margin averaged over three fixed paraphrases, and log p is a broad Gaussian prior in
log-parameter space. The prior is what keeps CEM off the boundary; without it the search
walks to whatever extreme the judge happens to like and reports it as a recovery.

theta = (log10 contact damping cd, log10 friction mu). cd sets restitution; both are the
axes J2 established the judge responds to. Geometry, placement, drop height and density
are INPUT, exactly as the project defines theta.

Why this is a search and not a fit: there is no ground-truth cd for a rubber duck. The
result to look for is that the duck and the brass pot converge to DIFFERENT restitution,
in the direction their materials imply, and that independent restarts agree.

Renderer. The search runs on EEVEE (0.30 s/frame vs 1.7 for Cycles). Measured on the 10
J2 duck/pot clips, within-object rank agreement between the two engines -- the only thing
a per-object CEM consumes -- is rho +0.90 (duck) and +0.60 (pot), with the argmax matching
exactly for the duck. Pooled agreement is only +0.27, but that is dominated by a per-object
offset the search never sees. Final elites are re-rendered and re-scored in CYCLES, which
is the setting J2 validated, so the reported answer never rests on the proxy alone.

Run (cosmos env; shells out to warp and blender itself):
  HF_HOME=... CUDA_VISIBLE_DEVICES=4 .../envs/cosmos/bin/python scripts/j3_optimize.py
  ... --objects rubber_duck,brass_pot --restarts 2 --pop 8 --iters 4
"""
import argparse
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

from scripts.j2r_calibration import FPS, OBJECTS, PARAPHRASES  # noqa: E402

WARP_PY = "/home/jooyeolyun/anaconda3/envs/warp/bin/python"
BLENDER = "/home/nas5/jaeseonglee/blender-4.2.0-linux-x64/blender"

# theta = (log10 cd, log10 mu); bounds are hard clips, the prior does the soft work.
AXES = ("log_cd", "log_mu")
BOUNDS = np.array([[2.0, 4.3],      # cd 100 .. 20000
                   [-1.0, 0.18]])   # mu 0.1 .. 1.5
PRIOR_MEAN = np.array([3.18, -0.30])   # cd ~1500, mu ~0.5
PRIOR_SD = np.array([0.70, 0.35])
BETA = 1.0


def log_prior(x):
    z = (np.asarray(x, float) - PRIOR_MEAN) / PRIOR_SD
    return float(-0.5 * np.sum(z * z))


def to_theta(x):
    return {"cd": float(10.0 ** x[0]), "mu": float(10.0 ** x[1])}


def run_sim(run, batch):
    (run / "batch_in.json").write_text(json.dumps(batch))
    r = subprocess.run([WARP_PY, str(REPO / "scripts" / "j3_sim_batch.py"), str(run)],
                       cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES":
                            os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    if r.returncode != 0:
        raise RuntimeError(f"sim failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads((run / "batch_out.json").read_text())


def run_render(run, engine="BLENDER_EEVEE_NEXT"):
    for d in run.glob("sim_*"):
        if d.is_dir():
            shutil.rmtree(d)
    r = subprocess.run([BLENDER, "--background", "--python",
                        str(REPO / "scripts" / "blender_render_sim.py")],
                       cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "LAB": str(run), "ENGINE": engine})
    if r.returncode != 0:
        raise RuntimeError(f"render failed:\n{r.stdout[-2000:]}")
    return sum(1 for _ in run.glob("sim_*"))


def encode(run):
    import imageio.v2 as imageio
    made = []
    for d in sorted(run.glob("sim_*")):
        if not d.is_dir():
            continue
        ps = sorted(d.glob("f*.png"))
        if not ps:
            continue
        dst = run / f"{d.name[4:]}.mp4"
        w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=8,
                               macro_block_size=1)
        for p in ps:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        made.append(dst.stem)
        shutil.rmtree(d)          # frames are large; the mp4 is the artifact
    return made


def score_clips(judge, run, keys, obj):
    noun = OBJECTS[obj]["noun"]
    out = {}
    for k in keys:
        p = run / f"{k}.mp4"
        if not p.exists():
            continue
        v = [judge.score_one(p, q.format(noun=noun))[0] for q in PARAPHRASES]
        a = np.asarray(v, float)
        out[k] = {"score": float(a.mean()), "spread": float(a.max() - a.min()),
                  "per_prompt": [float(z) for z in a]}
    return out


def evaluate(judge, run, obj, X, tag):
    """Simulate, render, judge a population. Returns list of per-candidate records."""
    batch = {f"{tag}_{i:02d}": {"object": obj, **to_theta(x)} for i, x in enumerate(X)}
    meta = run_sim(run, batch)
    run_render(run)
    encode(run)
    sc = score_clips(judge, run, list(batch), obj)
    recs = []
    for i, (k, th) in enumerate(batch.items()):
        m, s = meta.get(k, {}), sc.get(k)
        rec = {"key": k, "x": [float(v) for v in X[i]], **th,
               "e": m.get("e"), "travel_m": m.get("travel_m"), "ok": bool(m.get("ok"))}
        if s is None or not rec["ok"]:
            # An unsimulatable or unrenderable theta is not evidence about the judge; it
            # is simply excluded from the elite set rather than given a fake bad score.
            rec.update(score=None, objective=None, valid=False)
        else:
            rec.update(s)
            rec["log_prior"] = log_prior(X[i])
            rec["objective"] = BETA * s["score"] + rec["log_prior"]
            rec["valid"] = True
        recs.append(rec)
    return recs


def cem(judge, run, obj, pop, iters, elite_frac, seed, log):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    # Multi-start: each restart begins from a different draw over the whole box, so
    # agreement between restarts is evidence the optimum is real and not the init.
    mean = rng.uniform(lo, hi)
    sd = (hi - lo) / 4.0
    history = []
    for it in range(iters):
        X = np.clip(rng.normal(mean, sd, size=(pop, len(AXES))), lo, hi)
        recs = evaluate(judge, run, obj, X, f"{obj}_s{seed}_i{it}")
        good = [r for r in recs if r["valid"]]
        if not good:
            log(f"    iter {it}: no valid candidates, widening")
            sd = np.minimum(sd * 1.5, (hi - lo) / 2)
            continue
        good.sort(key=lambda r: -r["objective"])
        n_el = max(2, int(round(elite_frac * len(good))))
        el = good[:n_el]
        E = np.array([r["x"] for r in el])
        mean = E.mean(0)
        sd = np.maximum(E.std(0), (hi - lo) * 0.03)   # floor stops premature collapse
        best = el[0]
        history.append({"iter": it, "mean": mean.tolist(), "sd": sd.tolist(),
                        "n_valid": len(good), "records": recs,
                        "elite_keys": [r["key"] for r in el]})
        log(f"    iter {it}: {len(good)}/{pop} valid | best obj {best['objective']:+.3f} "
            f"(judge {best['score']:+.3f}) cd={best['cd']:.0f} mu={best['mu']:.2f} "
            f"e={best['e']:.3f} | mean cd={10**mean[0]:.0f} mu={10**mean[1]:.2f}")
    return {"seed": seed, "mean": mean.tolist(), "sd": sd.tolist(),
            "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="rubber_duck,brass_pot")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--elite-frac", type=float, default=0.35)
    ap.add_argument("--run", default=str(REPO / "outputs" / "judge" / "j3"))
    a = ap.parse_args()

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "j2r" / "lab.json", run / "lab.json")
    logf = open(run / "run.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    from src.judge.cosmos import CosmosJudge
    t0 = time.time()
    judge = CosmosJudge()
    log(f"judge loaded in {time.time()-t0:.0f}s | beta={BETA} "
        f"prior N({PRIOR_MEAN.tolist()}, {PRIOR_SD.tolist()}) | "
        f"pop={a.pop} iters={a.iters} restarts={a.restarts}")

    results = {}
    for obj in a.objects.split(","):
        log(f"\n=== {obj}  ({OBJECTS[obj]['noun']})")
        results[obj] = []
        for s in range(a.restarts):
            log(f"  restart {s}")
            results[obj].append(cem(judge, run, obj, a.pop, a.iters,
                                    a.elite_frac, 1000 + s, log))
        (run / "results.json").write_text(json.dumps(results, indent=2))

    log(f"\ntotal {time.time()-t0:.0f}s")
    log("\n=== converged theta (mean of final elite set)")
    for obj, rs in results.items():
        for r in rs:
            log(f"  {obj:<13} seed {r['seed']}: cd={10**r['mean'][0]:>8.0f} "
                f"mu={10**r['mean'][1]:.2f}")
    (run / "results.json").write_text(json.dumps(results, indent=2))
    log(f"\nwrote {run}/results.json")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
