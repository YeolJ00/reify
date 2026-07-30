"""Tracker-free screen: did anything in the clip actually move?

The first version of this screen measured the tracked centroid, and the tracker turned
out to be the thing being measured -- 11 of 12 baseball clips had points off-frame most
of the time, while a duck clip in which the duck visibly fell reported 6 px of motion
because the tracker stayed in mid-air. Point visibility does not detect a tracker locked
onto the background, so the failure is invisible in the tracker's own diagnostics.

This measures pixels instead. For each frame, the absolute difference from frame 0 inside
the staging region:

    changed_frac   fraction of region pixels that differ appreciably from frame 0
    travel_px      how far the CENTROID OF CHANGE moves across the clip
    settle_frac    fraction of the clip still changing at the end

None of it needs to know where the object is, which is exactly the property the tracker
lacked. It cannot say WHAT moved, so it is a screen, not a measurement -- if it says
nothing moved, there is nothing to measure and no probe design can rescue the clip.

Run: python scripts/screen_motion.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "screen"
ORDER = ["rest", "urgent", "airborne", "tipped"]
EQUILIBRIUM = {"rest": True, "urgent": True, "airborne": False, "tipped": False}
DIFF_THRESH = 18.0        # 8-bit levels; above sensor/compression noise in these clips
MIN_CHANGED = 0.004       # fraction of the region that must change to count as motion


def clip_motion(frames):
    """Motion statistics from frame differencing alone."""
    F = frames.astype(np.float32)
    n = len(F)
    ref = F[0]
    d = np.abs(F - ref).mean(axis=3)                     # (n, H, W) greyscale difference
    changed = d > DIFF_THRESH
    frac = changed.reshape(n, -1).mean(axis=1)

    H, W = d.shape[1], d.shape[2]
    yy, xx = np.mgrid[0:H, 0:W]
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    for t in range(n):
        m = changed[t]
        if m.sum() >= 40:                                # ignore specks
            cx[t] = xx[m].mean(); cy[t] = yy[m].mean()
    ok = np.isfinite(cx)
    if ok.sum() >= 3:
        P = np.stack([cx[ok], cy[ok]], 1)
        travel = float(np.hypot(*(P - P[0]).T).max())
    else:
        travel = 0.0
    tail = frac[int(0.75 * n):]
    return {"changed_frac": float(np.median(frac[1:])),
            "peak_frac": float(frac.max()),
            "travel_px": travel,
            "still_changing_at_end": float(np.median(tail)),
            "moved": bool(np.median(frac[1:]) > MIN_CHANGED)}


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    rows = {}
    for key, e in sorted(cfg["experiments"].items()):
        subj, variant = e["subject"], e["variant"]
        for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
            fr = np.load(f)["frames"]
            m = clip_motion(fr)
            m["seed"] = f.stem.split("seed")[1]
            rows.setdefault(subj, {}).setdefault(variant, []).append(m)

    print(f"threshold: a pixel counts as changed at {DIFF_THRESH:.0f}/255; a clip counts "
          f"as moving if >{MIN_CHANGED*100:.1f}% of pixels change\n")
    hdr = f"{'object':14s}" + "".join(f"{v:>20s}" for v in ORDER)
    print(hdr); print("-" * len(hdr))
    tally = {v: [0, 0] for v in ORDER}
    for subj, per in rows.items():
        cells = ""
        for v in ORDER:
            tk = per.get(v, [])
            if not tk:
                cells += f"{'—':>20s}"; continue
            nm = sum(t["moved"] for t in tk)
            tv = np.median([t["travel_px"] for t in tk])
            tally[v][0] += nm; tally[v][1] += len(tk)
            cells += f"{f'{nm}/{len(tk)} · {tv:5.0f}px':>20s}"
        print(f"{subj:14s}{cells}")
    print("-" * len(hdr))
    print(f"{'ALL':14s}" + "".join(f"{f'{tally[v][0]}/{tally[v][1]}':>20s}"
                                   for v in ORDER))
    print("\n(px = furthest the centroid of changed pixels travels, median over seeds)\n")

    eq = [0, 0]; ne = [0, 0]
    for v in ORDER:
        b = eq if EQUILIBRIUM[v] else ne
        b[0] += tally[v][0]; b[1] += tally[v][1]
    print("DOES THE INITIAL FRAME NEED TO SHOW NON-EQUILIBRIUM?")
    for nm, b in (("equilibrium     (rest, urgent)", eq),
                  ("non-equilibrium (airborne, tipped)", ne)):
        print(f"  {nm:38s} {b[0]:2d}/{b[1]:2d} clips moved "
              f"({100.0*b[0]/max(b[1],1):.0f}%)")

    print("\nBY OBJECT (median changed-pixel fraction across all four stagings)")
    for subj, per in rows.items():
        allt = [t for v in ORDER for t in per.get(v, [])]
        cf = np.median([t["changed_frac"] for t in allt])
        tv = np.median([t["travel_px"] for t in allt])
        nm = sum(t["moved"] for t in allt)
        print(f"  {subj:14s} {100*cf:5.2f}% of pixels change · centroid travels "
              f"{tv:5.0f}px · {nm}/{len(allt)} clips moved")

    (LAB / "screen_motion.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {LAB}/screen_motion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
