"""J2 KILL TEST: can the judge discriminate material character in OUR renders?

Everything downstream rests on this. If the judge cannot rank a silk-like rollout above a
tarp-like one when asked about silk, no optimizer over its score means anything, so this
runs before any optimizer exists.

Design:

* A grid over (stiffness x density x damping) spanning silk-like -> canvas-like ->
  tarp-like -> near-rigid. Each cell is simulated, checked for explosion, and rendered.
* Three material descriptions, each with a TARGET cell in log-parameter space. "Correct
  region" is defined by distance to that target rather than by hand-labelling every clip,
  so the labels cannot be quietly reshaped to fit the scores.
* Per material we report: Spearman rho between score and (negative) distance to target,
  AUC separating the nearest third from the farthest third, and the paraphrase spread.

PASS: correct-region clips clearly outrank wrong-material clips for all three materials.
FAIL: flat or non-monotone -> STOP, report, do not build J3.

Run (two passes, different envs):
  CUDA_VISIBLE_DEVICES=<g> .../envs/warp/bin/python   scripts/j2_calibration.py render
  HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/j2_calibration.py judge
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "judge" / "j2"
SECONDS = 3.0

# log-spaced grid: 4 stiffness x 3 density x 3 damping = 36 cells
KE = [2.0e2, 1.5e3, 8.0e3, 4.0e4]
MASS = [0.02, 0.05, 0.12]
KD = [2.0, 8.0, 25.0]

# Each material names a target cell. These are asserted priors about what the SIMULATOR
# should produce for that material, not fitted to the judge.
MATERIALS = {
    "light silk":          {"tri_ke": 2.0e2, "mass": 0.02, "tri_kd": 2.0},
    "heavy canvas":        {"tri_ke": 8.0e3, "mass": 0.05, "tri_kd": 8.0},
    "stiff plastic tarp":  {"tri_ke": 4.0e4, "mass": 0.12, "tri_kd": 25.0},
}
AXES = ("tri_ke", "mass", "tri_kd")


def log_distance(theta, target):
    """Normalised distance in log-parameter space; each axis scaled by its grid range."""
    spans = {"tri_ke": np.log(KE[-1] / KE[0]), "mass": np.log(MASS[-1] / MASS[0]),
             "tri_kd": np.log(KD[-1] / KD[0])}
    d = [(np.log(theta[a]) - np.log(target[a])) / spans[a] for a in AXES]
    return float(np.linalg.norm(d))


def cells():
    for ke, m, kd in itertools.product(KE, MASS, KD):
        yield {"tri_ke": ke, "mass": m, "tri_kd": kd}


def do_render():
    import yaml
    from src.render.clip import render_frames, theta_key, write_mp4
    from src.sim.rollout import FlagSim
    from src.sim.stability import is_exploded, substeps_for

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / "configs" / "flag.yaml"))
    cam = cfg["m4"]["camera"]
    rows, t_start = [], time.time()
    for i, th in enumerate(cells()):
        key = theta_key(th, "flag", cfg["seed"])
        path = OUT / f"cell_{key}.mp4"
        rec = {"theta": th, "clip": path.name, "key": key}
        if path.exists():
            rows.append({**rec, "cached": True}); continue
        c = dict(cfg); c["cloth"] = dict(cfg["cloth"]); c["sim"] = dict(cfg["sim"])
        c["cloth"].update(tri_ke=th["tri_ke"], tri_ka=th["tri_ke"],
                          tri_kd=th["tri_kd"], mass=th["mass"])
        c["sim"]["num_frames"] = int(SECONDS * cfg["sim"]["fps"])
        c["sim"]["substeps"] = substeps_for(th["tri_ke"], th["mass"], cfg["sim"]["fps"],
                                            kd=th["tri_kd"])
        sim = FlagSim(c); sim.rollout()
        X = sim.trajectory()
        blew, span = is_exploded(X)
        rec.update(substeps=c["sim"]["substeps"], span=round(span, 3), exploded=bool(blew))
        if blew:
            print(f"  [{i+1:2d}/36] EXPLODED span={span:.1f} {th}")
            rows.append(rec); continue
        frames = render_frames(sim, cam["eye"], cam["target"], width=640, height=480,
                               every=2)
        write_mp4(frames, path, fps=30)
        tip = X[:, -1, :]
        rec["tip_path_m"] = round(float(np.linalg.norm(np.diff(tip, 0 + 1, axis=0)
                                                       if False else np.diff(tip, axis=0),
                                                       axis=1).sum()), 3)
        rows.append(rec)
        print(f"  [{i+1:2d}/36] ke={th['tri_ke']:>7.0f} m={th['mass']:.2f} kd={th['tri_kd']:>5.1f}"
              f"  sub={rec['substeps']:>3d}  tip={rec['tip_path_m']:6.2f} m")
    (OUT / "cells.json").write_text(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if not r.get("exploded"))
    print(f"\n{ok}/{len(rows)} cells usable, {time.time()-t_start:.0f}s")
    print(f"wrote {OUT}/cells.json")
    return 0


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(pos, neg):
    """P(score of a correct-region clip > score of a wrong-material clip)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def do_judge():
    from src.judge.cosmos import CosmosJudge

    rows = json.loads((OUT / "cells.json").read_text())
    live = [r for r in rows if not r.get("exploded") and (OUT / r["clip"]).exists()]
    print(f"scoring {len(live)} clips x {len(MATERIALS)} materials")
    judge = CosmosJudge()
    t0 = time.time()
    for r in live:
        r["scores"] = {}
        for m in MATERIALS:
            r["scores"][m] = judge.score(str(OUT / r["clip"]), m)
    dt = time.time() - t0
    (OUT / "scored.json").write_text(json.dumps(live, indent=2))
    print(f"scored in {dt:.0f}s ({dt/(len(live)*len(MATERIALS)):.2f}s per pair)\n")
    report(live)
    return 0


def report(live):
    print(f"{'material':22s} {'spearman':>9s} {'AUC':>7s} {'spread':>8s} {'signal':>8s}  verdict")
    print("-" * 74)
    verdicts = {}
    for m, target in MATERIALS.items():
        d = np.array([log_distance(r["theta"], target) for r in live])
        s = np.array([r["scores"][m]["score"] for r in live])
        sp = np.array([r["scores"][m]["spread"] for r in live])
        rho = spearman(s, -d)                      # want positive: closer -> higher score
        k = max(len(live) // 3, 3)
        near = s[np.argsort(d)[:k]]                # correct region
        far = s[np.argsort(d)[-k:]]                # wrong material
        a = auc(near, far)
        signal = float(near.mean() - far.mean())
        ok = (a >= 0.70) and (rho > 0.25)
        verdicts[m] = {"spearman": rho, "auc": a, "spread": float(sp.mean()),
                       "signal": signal, "pass": bool(ok)}
        print(f"{m:22s} {rho:+9.3f} {a:7.3f} {sp.mean():8.3f} {signal:+8.3f}  "
              f"{'pass' if ok else 'FAIL'}")
    print("-" * 74)
    n_pass = sum(v["pass"] for v in verdicts.values())
    print(f"\nKILL TEST: {n_pass}/{len(MATERIALS)} materials pass")
    if n_pass == len(MATERIALS):
        print("VERDICT: PASS -- the judge discriminates material character. J3 may proceed.")
    else:
        print("VERDICT: FAIL -- do NOT build J3. Report and reconsider "
              "(stronger prompt, pairwise comparison, or a different direction).")
    (OUT / "verdict.json").write_text(json.dumps(verdicts, indent=2))
    return verdicts


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "render"
    if mode == "render":
        return do_render()
    if mode == "judge":
        return do_judge()
    if mode == "report":
        return 0 if report(json.loads((OUT / "scored.json").read_text())) else 1
    print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main())
