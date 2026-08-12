"""Joint SPSA fit of (friction, contact damping, density) over three probes.

WHAT EACH PARAMETER MEANS, and which probe can see it:

  mu       friction coefficient. How much grip between object and tabletop. Visible as the
           ANGLE at which a body starts to slide on a ramp: tan(angle) = mu. Mass-independent.
  cd       contact damping in the Hunt-Crossley law F = max(k*pen - cd*pen*v, 0). Higher cd
           removes more energy at impact, so it sets restitution -- how high and how often a
           dropped object bounces.
  rho      density, hence mass = density x mesh volume. Free fall does not depend on it, and
           contact depends only on the RATIOS k/m and cd/m, so a drop alone cannot separate
           density from damping. A collision can: momentum transfer to a partner of known
           mass sees mass directly.

That degeneracy is the reason this fit is joint. Fitting cd from a drop with density held
fixed -- which is what every earlier run in this project did, at rho = 600 for every object
-- is really fitting cd/m and calling it restitution.

  J(theta) = sum over probes of the judge's logit margin on that probe's clip

SPSA costs 2 evaluations per step regardless of theta dimension, so going from 1 parameter
to 3 is free; the cost is 2 x (number of probe scenes) renders per iteration. Both
evaluations of a step are the same object in the same scenes, so any per-object bias in the
judge cancels in y+ - y-.

Every iteration's clips are KEPT, so the fit can be watched rather than summarised.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/joint_fit.py
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
RUN = REPO / "outputs" / "judge" / "joint"
KEEP = RUN / "iters"
MODEL = "nvidia/Cosmos3-Super"

# WHICH PARAMETER EACH PROBE CAN SEE. A probe should only move what it has information
# about. Summing all three probes into one scalar and updating all of theta let the collide
# term -- measured loudest (|dy| 1.72) and least consistent (3/6) -- push mu, which it knows
# nothing about. Masked updates confine each probe's gradient to its own coordinates.
#   theta index:  0 = log10 mu, 1 = log10 cd, 2 = log10 rho
PROBE_MASK = {"tilt":    np.array([1.0, 0.0, 0.0]),   # slip angle       -> friction
              "drop":    np.array([0.0, 1.0, 0.0]),   # bounce           -> contact damping
              "collide": np.array([0.0, 0.0, 1.0]),   # momentum         -> density
              "spin":    np.array([1.0, 0.0, 0.0]),   # spin-down rate   -> friction
              "stack":   np.array([1.0, 0.0, 1.0]),   # topple angle     -> friction + mass
              "shove":   np.array([1.0, 0.0, 0.0]),   # stopping         -> friction
              "collide_heavy": np.array([0.0, 0.0, 1.0]),
              "collide_slow":  np.array([0.0, 0.0, 1.0])}

# WEIGHTS, measured rather than assumed. A probe that returns the same answer for every
# theta carries no information however sign-consistent it is. Fisher information for a
# yes/no judge is p(1-p) x (ds/dtheta)^2, so saturation and insensitivity both cost.
# Estimated from the yes-region sample (docs/results/probe_weights.json):
#     tilt 0.072   drop 0.430   collide 0.497
# Tilt was the MOST sign-consistent probe (5/6) and is the LEAST informative, because its
# mean p(yes) is 0.948 so p(1-p) = 0.047. Consistency and informativeness are different
# properties and equal weighting confused them. New probes start at the mean until measured.
_M = 0.33
PROBE_W = {"tilt": 0.072, "drop": 0.430, "collide": 0.497,
           "collide_heavy": 0.497, "collide_slow": 0.497,   # same family, measured value
           "spin": _M, "stack": _M, "shove": _M}

OBJECTS = {"brass_pot": 0.30, "wooden_bowl": -0.30}
NOUN = {"brass_pot": "brass pot", "wooden_bowl": "wooden bowl"}
# theta = (log10 mu, log10 cd, log10 rho)
X0 = np.array([np.log10(0.40), np.log10(2500.0), np.log10(1200.0)])
BOUNDS = np.array([[-0.90, 0.08], [3.00, 4.10], [2.20, 3.95]])
# a_k raised: the log-likelihood gradient is ~7x smaller than the raw margin's
N_ITER, A_GAIN, C_GAIN, ALPHA, GAMMA = 6, 0.40, 0.14, 0.602, 0.101

PROMPTS = {
    "spin": ("Look only at the {noun}. It has been spun on the spot. Is the way it keeps "
             "turning and slows to a stop consistent with what it is made of?"),
    "stack": ("Look only at the {noun}. Another object is balanced on top of it and the "
              "surface is tilting. Is the way the stack leans and gives way consistent "
              "with their weights and materials?"),
    "shove": ("Look only at the {noun}. It has been pushed across the table. Is the way it "
              "slides and comes to rest consistent with what it is made of?"),
    "tilt": ("Look only at the {noun}. The surface it rests on is tilting. Is the way it "
             "slides consistent with what it is made of? Consider its weight, hardness and "
             "grip on wood."),
    "drop": ("Look only at the {noun}. It is dropped onto the table. Is the way it bounces "
             "and settles consistent with what it is made of?"),
    "collide": ("Look only at the {noun}. It slides into another object and hits it. Is "
                "the way the two push each other consistent with their weights and "
                "materials?"),
    "collide_heavy": ("Look only at the {noun}. It slides into a heavier object. Is the way "
                      "the two push each other consistent with their weights?"),
    "collide_slow": ("Look only at the {noun}. It drifts slowly into another object and "
                     "nudges it. Is the way the two push each other consistent with their "
                     "weights?"),
}
TAIL = ("\nAssume the normal laws of physics. Base your answer on the events in the video "
        "and ignore the quality of the simulation engine.\n(A) Consistent\n(B) Inconsistent")


def loglik(s):
    """log p(yes | theta) = log sigmoid(s), NOT the raw logit margin s.

    Both are monotone in s so they share an argmax, but their gradients differ and for a
    gradient method that is what counts. d/ds of s is 1 always; d/ds of log sigmoid(s) is
    1 - sigmoid(s), which is 0.12 at s=+2 and 0.007 at s=+5. Measured over the previous
    joint fit, median s was +1.79 (p(yes)=0.86) with 39% above p=0.90 -- deep in saturation,
    where the true likelihood is flat and differences in s are carried by appearance and
    motion magnitude rather than physics.
    """
    return -np.log1p(np.exp(-np.clip(s, -30, 30)))


def to_theta(x):
    return {"mu": float(10 ** x[0]), "cd": float(10 ** x[1]), "rho": float(10 ** x[2])}


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


def evaluate(judge, xs_by_tag, it):
    spec = {tag: {o: {**to_theta(x[o]), "y": OBJECTS[o]} for o in OBJECTS}
            for tag, x in xs_by_tag.items()}
    (RUN / "joint_in.json").write_text(json.dumps(spec))
    gpu = pick_gpu()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    for step, args in (("sim", [WARP_PY, str(REPO / "scripts" / "joint_sim.py"), str(RUN)]),):
        r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
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
    c = subprocess.run([WARP_PY, str(REPO / "scripts" / "joint_sim.py"), str(RUN), "crop"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    if c.returncode != 0:
        raise RuntimeError(f"crop failed:\n{c.stdout[-1200:]}\n{c.stderr[-1200:]}")

    crops = json.loads((RUN / "crops.json").read_text())
    scores = {}
    dst = KEEP / f"it{it:02d}"
    dst.mkdir(parents=True, exist_ok=True)
    for rec in crops:
        o, probe, tag = rec["object"], rec["probe"], rec["tag"]
        if o not in OBJECTS:
            continue
        p = RUN / rec["clip"]
        if not p.exists():
            continue
        q = PROMPTS[probe].format(noun=NOUN[o]) + TAIL
        s = float(judge.score(p, q))
        # one simulation, several viewpoints: average so a theta must read plausible from
        # every angle, not just the one the camera happened to be at
        scores.setdefault(tag, {}).setdefault(o, {}).setdefault(probe, []).append(s)
        shutil.copy(p, dst / rec["clip"])     # keep every iteration's clips
    for tag in scores:
        for o in scores[tag]:
            scores[tag][o] = {pr: float(np.mean(v)) for pr, v in scores[tag][o].items()}
    return scores


def main():
    from src.judge.plausibility import PlausibilityJudge
    RUN.mkdir(parents=True, exist_ok=True); KEEP.mkdir(exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "g0c" / "lab.json", RUN / "lab.json")
    print(__doc__.split("Run:")[0])
    judge = PlausibilityJudge(model_id=MODEL, load_in_8bit=True)

    x = {o: X0.copy() for o in OBJECTS}
    rng = np.random.default_rng(0)
    hist, t0 = [], time.time()
    for k in range(N_ITER):
        a_k = A_GAIN / (0.1 * N_ITER + k + 1) ** ALPHA
        c_k = C_GAIN / (k + 1) ** GAMMA
        delta = {o: rng.choice([-1.0, 1.0], size=3) for o in OBJECTS}
        xp = {o: np.clip(x[o] + c_k * delta[o], BOUNDS[:, 0], BOUNDS[:, 1]) for o in OBJECTS}
        xm = {o: np.clip(x[o] - c_k * delta[o], BOUNDS[:, 0], BOUNDS[:, 1]) for o in OBJECTS}
        sc = evaluate(judge, {"plus": xp, "minus": xm}, k)
        rec = {"iter": k, "objects": {}}
        for o in OBJECTS:
            sp, sm = sc.get("plus", {}).get(o, {}), sc.get("minus", {}).get(o, {})
            probes = sorted(set(sp) & set(sm))
            if not probes:
                continue
            # masked, per-probe, on the LOG-LIKELIHOOD rather than the raw margin
            g = np.zeros(3)
            for pr in probes:
                d = loglik(sp[pr]) - loglik(sm[pr])
                g += PROBE_W.get(pr, _M) * PROBE_MASK[pr] * d / (2.0 * c_k * delta[o])
            yp, ym = sum(sp[p] for p in probes), sum(sm[p] for p in probes)
            x[o] = np.clip(x[o] + a_k * g, BOUNDS[:, 0], BOUNDS[:, 1])
            rec["objects"][o] = {"theta": to_theta(x[o]),
                                 "theta_plus": to_theta(xp[o]),
                                 "theta_minus": to_theta(xm[o]),
                                 "per_probe_plus": sp, "per_probe_minus": sm,
                                 "y_plus": yp, "y_minus": ym, "dy": yp - ym,
                                 "probes": probes}
            th = to_theta(x[o])
            print(f"  it{k} {o:<12} dy {yp-ym:+6.2f} | " +
                  " ".join(f"{p}{sp[p]-sm[p]:+5.2f}" for p in probes) +
                  f" -> mu {th['mu']:.3f} cd {th['cd']:.0f} rho {th['rho']:.0f}")
        hist.append(rec)
        print(f"  iter {k}/{N_ITER}  {(time.time()-t0)/60:.1f} min")
        (RUN / "results.json").write_text(json.dumps(
            {"history": hist, "theta": {o: to_theta(x[o]) for o in OBJECTS}}, indent=2))
    print("\nfinal:")
    for o in OBJECTS:
        th = to_theta(x[o])
        print(f"  {o:<13} mu {th['mu']:.3f}  cd {th['cd']:.0f}  rho {th['rho']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
