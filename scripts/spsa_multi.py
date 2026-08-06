"""SPSA over several objects and several seeds, in lockstep.

Purpose: the single paired run gave the correct direction (duck bouncier than pot, +0.125)
with a clean rule-4 result, but the separation rested on one evaluation -- the duck's 2nd
and 3rd best sat at e~0.15, indistinguishable from the pot's best. Seeds decide whether
that ordering is real or noise, and more objects turn one contrast into a ranking that can
be checked against material sense.

Why lockstep and batched. One evaluation is a Newton rollout, a Cycles render and a 33B
judge forward, and Blender's ~25 s startup dominates a ~35 s single-clip render. Running
every (object, seed) pair at the same SPSA iteration lets all of their perturbations --
2 per pair -- go into ONE Blender launch. With 4 objects x 3 seeds that is 24 clips per
launch instead of 24 launches, roughly halving wall clock.

SPSA state per pair is independent; only the rendering is shared.

Prediction being tested, from the earlier calibration and the single fit:
  if the judge's prior conditions on material, the recovered restitution should ORDER as
  rubber duck > baseball > ceramic vase ~ brass pot, consistently across seeds.
  if the ordering flips with seed, the single run was noise.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=6 .../envs/cosmos/bin/python scripts/spsa_multi.py
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

WARP_PY = "/home/jooyeolyun/anaconda3/envs/warp/bin/python"
BLENDER = "/home/nas5/jaeseonglee/blender-4.2.0-linux-x64/blender"
RUN = REPO / "outputs" / "judge" / "spsa_multi"
KEEP = RUN / "clips"           # final/best clip per (object, seed), kept for review
MODEL = "nvidia/Cosmos3-Super"

PROMPT = ("Is the way this object moves consistent with what it is made of? Consider its "
          "weight, hardness and how much such a material should bounce. Assume the normal "
          "laws of physics.\nYour answer should be based on the events in the video and "
          "ignore the quality of the simulation engine.\n(A) Consistent\n(B) Inconsistent")

# material prior, for reading the result only -- never fed to the optimiser
OBJECTS = {"rubber_duck": "soft rubber, should bounce most",
           "baseball": "leather and wound yarn, moderate",
           "ceramic_vase": "hard and brittle, little bounce",
           "brass_pot": "heavy metal, least bounce"}
SEEDS = [0, 1, 2]
X0 = np.array([np.log10(2500.0), np.log10(0.5)])
BOUNDS = np.array([[3.00, 4.10], [-0.80, 0.05]])
N_ITER = 10
A_GAIN, C_GAIN, ALPHA, GAMMA = 0.05, 0.15, 0.602, 0.101


def to_theta(x):
    return {"cd": float(10.0 ** x[0]), "mu": float(10.0 ** x[1])}


def pick_render_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    mine = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    best, bf = None, 0
    for line in out.strip().splitlines():
        i, u, t = [z.strip() for z in line.split(",")]
        if i == mine:
            continue
        f = int(t) - int(u)
        if f > bf:
            best, bf = i, f
    if best is None or bf < 6000:
        raise RuntimeError(f"no render GPU with >=6 GB free (best {bf} MiB)")
    return best


def render_and_score(judge, batch, keep=()):
    """batch: {key: {object, cd, mu}} -> {key: record}. One sim + one Blender launch."""
    from src.render.motion_budget import sample_frames, check
    import imageio.v2 as imageio

    (RUN / "batch_in.json").write_text(json.dumps(batch))
    gpu = pick_render_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    r = subprocess.run([WARP_PY, str(REPO / "scripts" / "spsa_sim_batch.py"), str(RUN)],
                       cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"sim failed:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    meta = json.loads((RUN / "batch_out.json").read_text())
    for d in RUN.glob("sim_*"):
        if d.is_dir():
            shutil.rmtree(d)
    for p in RUN.glob("*.mp4"):
        p.unlink()
    rr = subprocess.run([BLENDER, "--background", "--python",
                         str(REPO / "scripts" / "blender_render_sim.py")],
                        cwd=REPO, capture_output=True, text=True,
                        env={**env, "LAB": str(RUN)})
    if rr.returncode != 0:
        raise RuntimeError(f"render failed:\n{rr.stdout[-1200:]}")

    out = {}
    for k, th in batch.items():
        d = RUN / f"sim_{k}"
        ps = sorted(d.glob("f*.png")) if d.is_dir() else []
        if not meta.get(k, {}).get("ok") or not ps:
            out[k] = None
            continue
        dst = RUN / f"{k}.mp4"
        w = imageio.get_writer(str(dst), fps=24, codec="libx264", quality=8,
                               macro_block_size=1)
        for p in ps:
            w.append_data(imageio.imread(p)[..., :3])
        w.close()
        ok, frac, _ = check(dst)
        v = sample_frames(dst, 12).astype(np.float32)
        rec = {"key": k, **th, **meta[k], "guard_ok": bool(ok),
               "motion_fraction": round(frac, 3),
               "M": round(float(np.abs(np.diff(v, axis=0)).mean()), 3)}
        rec["s"] = float(judge.score(dst, PROMPT))
        out[k] = rec
        if k in keep:
            KEEP.mkdir(parents=True, exist_ok=True)
            shutil.copy(dst, KEEP / f"{k}.mp4")
        shutil.rmtree(d)
    return out


def main():
    from src.judge.plausibility import PlausibilityJudge

    RUN.mkdir(parents=True, exist_ok=True)
    KEEP.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "g0c" / "lab.json", RUN / "lab.json")
    print(__doc__.split("Run:")[0])
    t0 = time.time()
    judge = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"judge loaded in {time.time()-t0:.0f}s")

    pairs = [(o, s) for o in OBJECTS for s in SEEDS]
    state = {p: {"x": X0.copy(), "rng": np.random.default_rng(1000 + p[1]),
                 "sd": (BOUNDS[:, 1] - BOUNDS[:, 0]) / 4.0, "trace": []} for p in pairs}
    print(f"{len(pairs)} (object, seed) pairs x {N_ITER} iterations "
          f"x 2 evals = {len(pairs)*N_ITER*2} evaluations\n")

    t0 = time.time()
    for k in range(N_ITER):
        a_k = A_GAIN / (0.1 * N_ITER + k + 1) ** ALPHA
        c_k = C_GAIN / (k + 1) ** GAMMA
        batch, plan = {}, {}
        for p in pairs:
            st = state[p]
            delta = st["rng"].choice([-1.0, 1.0], size=2)
            xp = np.clip(st["x"] + c_k * delta, BOUNDS[:, 0], BOUNDS[:, 1])
            xm = np.clip(st["x"] - c_k * delta, BOUNDS[:, 0], BOUNDS[:, 1])
            kp, km = f"{p[0]}_s{p[1]}_i{k:02d}_p", f"{p[0]}_s{p[1]}_i{k:02d}_m"
            batch[kp] = {"object": p[0], **to_theta(xp)}
            batch[km] = {"object": p[0], **to_theta(xm)}
            plan[p] = (delta, xp, xm, kp, km)
        recs = render_and_score(judge, batch)
        for p in pairs:
            delta, xp, xm, kp, km = plan[p]
            rp, rm = recs.get(kp), recs.get(km)
            yp = rp["s"] if rp else -10.0
            ym = rm["s"] if rm else -10.0
            for r in (rp, rm):
                if r:
                    state[p]["trace"].append(r)
            g = (yp - ym) / (2.0 * c_k * delta)
            state[p]["x"] = np.clip(state[p]["x"] + a_k * g,
                                    BOUNDS[:, 0], BOUNDS[:, 1])
        el = time.time() - t0
        print(f"iter {k:>2}/{N_ITER}  a_k={a_k:.4f} c_k={c_k:.4f}  "
              f"{2*len(pairs)} evals  elapsed {el/60:.1f} min  "
              f"eta {el/(k+1)*(N_ITER-k-1)/60:.0f} min")
        json.dump({f"{o}|{s}": {"x": state[(o, s)]["x"].tolist(),
                                "theta": to_theta(state[(o, s)]["x"]),
                                "trace": state[(o, s)]["trace"]}
                   for o, s in pairs}, open(RUN / "results.json", "w"), indent=2)

    # final clip per pair, at the converged theta, kept for review
    final = {f"{o}_s{s}_FINAL": {"object": o, **to_theta(state[(o, s)]["x"])}
             for o, s in pairs}
    fr = render_and_score(judge, final, keep=set(final))
    for o, s in pairs:
        state[(o, s)]["final_rec"] = fr.get(f"{o}_s{s}_FINAL")

    res = {f"{o}|{s}": {"object": o, "seed": s,
                        "theta": to_theta(state[(o, s)]["x"]),
                        "final": state[(o, s)].get("final_rec"),
                        "trace": state[(o, s)]["trace"]} for o, s in pairs}
    json.dump(res, open(RUN / "results.json", "w"), indent=2)

    print(f"\n{'object':<15} " + " ".join(f"{'seed '+str(s):>13}" for s in SEEDS) +
          f" {'mean e':>8}  prior")
    print("-" * 84)
    order = []
    for o in OBJECTS:
        es, cells = [], []
        for s in SEEDS:
            f = state[(o, s)].get("final_rec")
            if f and f.get("e") is not None:
                es.append(f["e"])
                cells.append(f"cd{f['cd']:>5.0f} e{f['e']:.2f}")
            else:
                cells.append("     --     ")
        m = float(np.mean(es)) if es else float("nan")
        order.append((o, m))
        print(f"{o:<15} " + " ".join(f"{c:>13}" for c in cells) +
              f" {m:>8.3f}  {OBJECTS[o]}")
    print("\nrecovered restitution, ranked:")
    for o, m in sorted(order, key=lambda z: -z[1]):
        print(f"  {o:<15} {m:.3f}")
    print("\nexpected if the prior conditions on material:")
    print("  rubber_duck > baseball > ceramic_vase ~ brass_pot")
    print(f"\nclips kept in {KEEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
