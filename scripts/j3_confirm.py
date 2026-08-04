"""J3 confirmation: re-score the CEM's answer in CYCLES, the engine J2 validated.

The search ran on EEVEE for speed. This sweeps cd at each object's converged friction and
renders the SAME thetas in BOTH engines, so any disagreement is attributable to the
renderer rather than to theta, sampling noise, or the prompt.

If Cycles puts the optimum somewhere else, the CEM optimised a proxy artefact and the J3
answer does not stand -- which is the point of running this rather than reporting the
converged numbers directly.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=4 .../envs/cosmos/bin/python scripts/j3_confirm.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.j2r_calibration import OBJECTS, PARAPHRASES  # noqa: E402
from scripts.j3_optimize import encode, run_render, run_sim, score_clips  # noqa: E402

RUN = REPO / "outputs" / "judge" / "j3conf"
CDS = [300.0, 600.0, 1200.0, 2500.0, 4000.0, 8000.0]
ENGINES = {"eevee": "BLENDER_EEVEE_NEXT", "cycles": "CYCLES"}


def main():
    res = json.loads((REPO / "outputs" / "judge" / "j3" / "results.json").read_text())
    # friction from the CEM, averaged over restarts -- hold it fixed and sweep cd
    mu = {o: float(np.mean([10 ** r["mean"][1] for r in rs])) for o, rs in res.items()}
    conv = {o: float(np.mean([10 ** r["mean"][0] for r in rs])) for o, rs in res.items()}
    RUN.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "outputs" / "judge" / "j2r" / "lab.json", RUN / "lab.json")
    print("CEM converged to:")
    for o in res:
        print(f"  {o:<13} cd={conv[o]:.0f}  mu={mu[o]:.2f}")

    batch = {f"{o}_cd{int(cd)}": {"object": o, "cd": cd, "mu": mu[o]}
             for o in res for cd in CDS}
    meta = run_sim(RUN, batch)
    print(f"simulated {len(batch)}, usable {sum(1 for v in meta.values() if v['ok'])}")

    from src.judge.cosmos import CosmosJudge
    judge = CosmosJudge()

    out = {}
    for tag, eng in ENGINES.items():
        t0 = time.time()
        run_render(RUN, engine=eng)
        encode(RUN)
        print(f"  {tag}: rendered in {time.time()-t0:.0f}s")
        for o in res:
            keys = [k for k in batch if batch[k]["object"] == o]
            sc = score_clips(judge, RUN, keys, o)
            for k, v in sc.items():
                out.setdefault(k, {"object": o, "cd": batch[k]["cd"],
                                   "mu": batch[k]["mu"], "e": meta[k]["e"],
                                   "travel_m": meta[k]["travel_m"]})[tag] = v["score"]
        # keep the cycles clips for the artifact; clear eevee before the next pass
        if tag != list(ENGINES)[-1]:
            for p in RUN.glob("*.mp4"):
                p.unlink()

    (RUN / "confirm.json").write_text(json.dumps(out, indent=2))

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.corrcoef(np.argsort(np.argsort(a)),
                                 np.argsort(np.argsort(b)))[0, 1])

    print(f"\n{'cell':<22} {'e':>6} {'eevee':>8} {'cycles':>8} {'diff':>7}")
    print("-" * 56)
    for o in res:
        rs = sorted([v | {"key": k} for k, v in out.items() if v["object"] == o],
                    key=lambda r: r["cd"])
        for r in rs:
            print(f"{r['key']:<22} {r['e']:>6.3f} {r.get('eevee', float('nan')):>+8.3f} "
                  f"{r.get('cycles', float('nan')):>+8.3f} "
                  f"{r.get('eevee', 0)-r.get('cycles', 0):>+7.3f}")
        ee = [r["eevee"] for r in rs]
        cy = [r["cycles"] for r in rs]
        e = [r["e"] for r in rs]
        be, bc = rs[int(np.argmax(ee))], rs[int(np.argmax(cy))]
        print(f"  {o}: rho(eevee,cycles) = {sp(ee, cy):+.3f} | "
              f"rho(cycles,e) = {sp(cy, e):+.3f}")
        print(f"  {o}: argmax eevee cd={be['cd']:.0f} (e={be['e']:.3f})  "
              f"argmax cycles cd={bc['cd']:.0f} (e={bc['e']:.3f})\n")
    print(f"wrote {RUN}/confirm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
