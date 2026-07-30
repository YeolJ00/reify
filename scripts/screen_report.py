"""Did the generator actually move the object? One number per staging variant.

This is the screen that should have run before any probe was designed. It asks nothing
about materials -- only whether the requested motion is present at all, using the same
significance test the estimator uses: a fitted velocity that is not distinguishable from
zero means the motion did not happen.

Reports, per variant, the subject's net displacement in object-widths and whether the
initial speed is significant. The comparison that matters is between stagings whose
initial frame shows a static EQUILIBRIUM (rest, urgent) and ones that do not (airborne,
tipped): if only the latter move, no prompt will fix it and the staging must change.

Run: python scripts/screen_report.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.motion.observables import (SIG_K, _direction, _valid,  # noqa: E402
                                    max_excursion, net_travel, significant, velocity)

LAB = REPO / "outputs" / "scene" / "screen"
EQUILIBRIUM = {"rest": True, "urgent": True, "airborne": False, "tipped": False}
ORDER = ["rest", "urgent", "airborne", "tipped"]


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    rows = {}
    for key, e in sorted(cfg["experiments"].items()):
        subj, variant = e["subject"], e["variant"]
        sp = seeds[key]["subject"]["size_px"]
        per = []
        for t in sorted(LAB.glob(f"trk_{key}_seed*.npz")):
            d = np.load(t)
            if "subject_cen" not in d:
                continue
            cen = d["subject_cen"]
            if np.isnan(cen[:, 0]).mean() > 0.5:
                continue
            xy, ok = _valid(cen)
            if ok.sum() < 8:
                continue
            dirv = _direction(xy, ok)
            v0, sv0 = velocity(xy, ok, 0, min(12, len(xy) - 1), dirv)
            travel = max(net_travel(xy), max_excursion(xy)) / max(sp, 1e-6)
            per.append({"seed": t.stem.split("seed")[1], "travel": travel,
                        "v0": v0, "v0_se": sv0, "moved": significant(v0, sv0)})
        rows.setdefault(subj, {})[variant] = per

    print(f"significance: a velocity counts as motion at {SIG_K:.0f} standard errors\n")
    hdr = f"{'object':14s}" + "".join(f"{v:>22s}" for v in ORDER)
    print(hdr); print("-" * len(hdr))
    tally = {v: [0, 0] for v in ORDER}
    for subj, per in rows.items():
        cells = ""
        for v in ORDER:
            takes = per.get(v, [])
            if not takes:
                cells += f"{'—':>22s}"; continue
            nmov = sum(t["moved"] for t in takes)
            tv = np.median([t["travel"] for t in takes])
            tally[v][0] += nmov; tally[v][1] += len(takes)
            cells += f"{f'{nmov}/{len(takes)} moved · {tv:.2f}w':>22s}"
        print(f"{subj:14s}{cells}")
    print("-" * len(hdr))
    print(f"{'ALL':14s}" + "".join(
        f"{f'{tally[v][0]}/{tally[v][1]}':>22s}" for v in ORDER))
    print("\n(w = object-widths of net travel, median over seeds)\n")

    print("EQUILIBRIUM IN THE INITIAL FRAME vs MOTION PRODUCED")
    eq = [0, 0]; ne = [0, 0]
    for v in ORDER:
        b = eq if EQUILIBRIUM[v] else ne
        b[0] += tally[v][0]; b[1] += tally[v][1]
    for nm, b in (("frame shows equilibrium  (rest, urgent)", eq),
                  ("frame shows non-equilibrium (airborne, tipped)", ne)):
        pct = 100.0 * b[0] / max(b[1], 1)
        print(f"  {nm:48s} {b[0]:2d}/{b[1]:2d} takes moved  ({pct:.0f}%)")

    # detail per variant, so a weak manipulation is visible rather than averaged away
    print("\nper-take detail")
    for subj, per in rows.items():
        for v in ORDER:
            for t in per.get(v, []):
                mark = "moved" if t["moved"] else "STATIC"
                print(f"  {subj:13s} {v:9s} seed {t['seed']:<3s} {mark:6s} "
                      f"v0={t['v0']:6.2f}±{t['v0_se']:.2f} px/frame  "
                      f"travel={t['travel']:.2f}w")

    (LAB / "screen.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {LAB}/screen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
