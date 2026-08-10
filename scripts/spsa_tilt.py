"""SPSA on friction, via the tilt probe, several objects at once.

Two things make this the right shape, and I had missed both:

1. NO SWEEP. SPSA perturbs and follows a gradient estimate; sweeping is what it replaces.
   A sweep costs O(levels) evaluations per object; SPSA costs 2 per iteration regardless.

2. THE OBJECT CONFOUND CANCELS FOR FREE. The previous run gave each object exactly one mu,
   so friction and object identity varied together and rho(score, mu) = +0.400 was
   indistinguishable from "the judge likes brass pots" -- the pot also scored +2.885 in the
   DROP scene, where friction never varied. SPSA's estimate uses (y_plus - y_minus) for the
   SAME object one perturbation apart, so any per-object offset subtracts out by
   construction. That is the same trick pairwise scoring was using, obtained here for free.

Batching: one scene holds every object at its mu_plus and another every object at its
mu_minus, so an iteration costs TWO renders no matter how many objects are being fitted.

Objects are sliders only. Measured earlier: a sphere rolls and has no slip threshold
(baseball corr +0.32 in mu against +0.92 for the book), and a tall narrow object topples
instead of sliding (the ceramic vase moves at 8.5 deg despite mu=0.80), which is the
centre-of-mass probe rather than friction.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/spsa_tilt.py
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
RUN = REPO / "outputs" / "judge" / "spsa_tilt"
MODEL = "nvidia/Cosmos3-Super"

# sliders only, well separated so crops do not overlap (measured IoU 0.08 at 0.60 m apart)
OBJECTS = {"book": -0.30, "brass_pot": 0.30, "wooden_bowl": 0.00}
X0 = np.log10(0.40)                      # same start for every object
BOUNDS = (np.log10(0.08), np.log10(1.20))
N_ITER, A_GAIN, C_GAIN, ALPHA, GAMMA = 8, 0.09, 0.16, 0.602, 0.101

# The object is NAMED now. Clips keep the whole scene -- table edge, incline, neighbours --
# because a tight crop deleted the very slope the sliding is judged against, so the question
# has to say which object it means. Naming leans on the capability measured reliable in this
# model (object and material identification, 4/4 open-ended) rather than on pixel isolation.
NOUN = {"book": "hardcover book set", "brass_pot": "brass pot",
        "wooden_bowl": "wooden bowl"}
PROMPT = ("Look only at the {noun} in this video. The surface it rests on is tilting. Is the "
          "way it slides consistent with what it is made of? Consider its weight, hardness "
          "and how much grip such a material should have on wood. Assume the normal laws of "
          "physics.\nYour answer should be based on the events in the video and ignore the "
          "quality of the simulation engine.\n(A) Consistent\n(B) Inconsistent")


def pick_gpu():
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
        raise RuntimeError(f"no render GPU free (best {bf} MiB)")
    return best


def build_and_score(judge, mus_by_scene):
    """mus_by_scene: {scene_name: {object: mu}} -> {scene: {object: score}}.

    One sim call and one Blender call per scene, then per-object crops.
    """
    spec = {s: {o: {"mu": float(m), "y": OBJECTS[o]} for o, m in d.items()}
            for s, d in mus_by_scene.items()}
    (RUN / "spsa_in.json").write_text(json.dumps(spec))
    gpu = pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    r = subprocess.run([WARP_PY, str(REPO / "scripts" / "spsa_tilt_sim.py"), str(RUN)],
                       cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"sim failed:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    for d in RUN.glob("sim_*"):
        if d.is_dir():
            shutil.rmtree(d)
    for p in RUN.glob("*.mp4"):
        p.unlink()
    rr = subprocess.run([BLENDER, "--background", "--python",
                         str(REPO / "scripts" / "blender_render_scene.py")],
                        cwd=REPO, capture_output=True, text=True,
                        env={**env, "LAB": str(RUN)})
    if rr.returncode != 0:
        raise RuntimeError(f"render failed:\n{rr.stdout[-1200:]}")
    c = subprocess.run([WARP_PY, str(REPO / "scripts" / "spsa_tilt_sim.py"), str(RUN),
                        "crop"], cwd=REPO, capture_output=True, text=True, env=env)
    if c.returncode != 0:
        raise RuntimeError(f"crop failed:\n{c.stdout[-1200:]}\n{c.stderr[-1200:]}")
    crops = json.loads((RUN / "crops.json").read_text())
    out = {}
    for rec in crops:
        p = RUN / rec["clip"]
        if not p.exists():
            continue
        out.setdefault(rec["scene"], {})[rec["object"]] = {
            "s": float(judge.score(p, PROMPT.format(noun=NOUN[rec["object"]]))),
            "mu": rec["mu"],
            "onset_deg": rec.get("onset_deg"), "M": rec.get("motion_in_crop")}
    return out


def main():
    from src.judge.plausibility import PlausibilityJudge

    RUN.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "g0c" / "lab.json", RUN / "lab.json")
    print(__doc__.split("Run:")[0])
    t0 = time.time()
    judge = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)
    print(f"judge loaded in {time.time()-t0:.0f}s\n")

    x = {o: X0 for o in OBJECTS}
    rng = np.random.default_rng(0)
    hist = []
    t0 = time.time()
    for k in range(N_ITER):
        a_k = A_GAIN / (0.1 * N_ITER + k + 1) ** ALPHA
        c_k = C_GAIN / (k + 1) ** GAMMA
        delta = {o: float(rng.choice([-1.0, 1.0])) for o in OBJECTS}
        plus = {o: 10 ** float(np.clip(x[o] + c_k * delta[o], *BOUNDS)) for o in OBJECTS}
        minus = {o: 10 ** float(np.clip(x[o] - c_k * delta[o], *BOUNDS)) for o in OBJECTS}
        sc = build_and_score(judge, {"plus": plus, "minus": minus})
        rec = {"iter": k, "a_k": a_k, "c_k": c_k, "objects": {}}
        for o in OBJECTS:
            yp = sc.get("plus", {}).get(o, {}).get("s")
            ym = sc.get("minus", {}).get(o, {}).get("s")
            if yp is None or ym is None:
                rec["objects"][o] = {"mu": 10 ** x[o], "skipped": True}
                continue
            g = (yp - ym) / (2.0 * c_k * delta[o])
            x[o] = float(np.clip(x[o] + a_k * g, *BOUNDS))
            rec["objects"][o] = {"mu_plus": plus[o], "mu_minus": minus[o],
                                 "y_plus": yp, "y_minus": ym, "grad": g,
                                 "mu": 10 ** x[o]}
            print(f"  it{k} {o:<13} mu+ {plus[o]:.3f} s{yp:+.2f} | "
                  f"mu- {minus[o]:.3f} s{ym:+.2f} | dy {yp-ym:+.2f} -> mu {10**x[o]:.3f}")
        hist.append(rec)
        el = time.time() - t0
        print(f"  iter {k}/{N_ITER}  {el/60:.1f} min  eta {el/(k+1)*(N_ITER-k-1)/60:.0f} min")
        (RUN / "results.json").write_text(json.dumps(
            {"history": hist, "mu": {o: 10 ** v for o, v in x.items()}}, indent=2))

    print(f"\n{'object':<14} {'start mu':>9} {'final mu':>9}")
    for o in OBJECTS:
        print(f"{o:<14} {10**X0:>9.3f} {10**x[o]:>9.3f}")
    print("\nEach object's gradient came from a difference between two clips of THAT object,")
    print("so a per-object offset in the judge cannot drive the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
