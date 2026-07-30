"""Re-audit every lab take with an appearance-based tracker instead of point tracking.

The published audit ("68 of 93 takes contained no measurable motion") was built on
CoTracker centroids, and 38% of those clips later failed a tracker-health check -- with
the worst failure mode, points locked confidently onto the background, invisible to that
check. This re-runs the same question with an independent instrument (NCC patch
matching, src/motion/patch_track.py) and reports where the two disagree.

Each take lands in one of three states:

    MOVES        displacement well above the noise floor, appearance preserved
    STATIC       no displacement, appearance preserved -- genuinely nothing happened
    DEGRADED     appearance collapses; the object stopped being the object, so no
                 rigid-body theta explains it and it is not a physics measurement at all

That third state did not exist in the previous audit, which could only say
"usable" or "no motion", and so filed asset breakdown under no-motion.

Run: LAB=<dir> python scripts/audit_pixel.py        (warp env, no GPU)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.motion.patch_track import track_patch  # noqa: E402

LAB = Path(os.environ.get("LAB") or (REPO / "outputs" / "scene" / "fulllab"))
# Displacement is measured END TO END and in object-widths, not as the maximum
# excursion in pixels. The max spikes whenever the object transiently breaks up --
# the ceramic vase splits into two blobs mid-clip and re-forms, which scored 38px of
# "motion" for an object that visually never left its spot. Same failure as the old
# path-length gate: a statistic that accumulates transients is not a displacement.
MOVE_WIDTHS = 0.30
NCC_OK = 0.55        # below this the patch no longer matches its own object
SEARCH = 120


def classify(r, size_px):
    if r is None:
        return "NOTRACK", "patch could not be matched at all"
    if r["ncc_end"] < NCC_OK or r["ncc_median"] < NCC_OK:
        return "DEGRADED", (f"appearance collapsed (ncc {r['ncc_median']:.2f} median, "
                            f"{r['ncc_end']:.2f} at the end)")
    w = r["end_px"] / max(size_px, 1e-6)
    if w < MOVE_WIDTHS:
        return "STATIC", (f"ended {r['end_px']:.0f}px from where it began "
                          f"({w:.2f} widths), appearance intact")
    return "MOVES", (f"{r['end_px']:.0f}px end to end ({w:.2f} widths), "
                     f"ncc {r['ncc_median']:.2f}")


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    counts, per_kind, detail = {}, {}, []

    for key, e in sorted(cfg["experiments"].items()):
        s = seeds.get(key, {}).get("subject")
        if not s:
            continue
        kind = e.get("kind", e.get("variant", "?"))
        for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
            fr = np.load(f)["frames"]
            r = track_patch(fr, s["u"], s["v"], half=max(s["size_px"] * 0.45, 10),
                            search=SEARCH)
            state, why = classify(r, s["size_px"])
            counts[state] = counts.get(state, 0) + 1
            per_kind.setdefault(kind, {}).setdefault(state, 0)
            per_kind[kind][state] += 1
            detail.append({"clip": f.stem.replace("vid_", ""), "kind": kind,
                           "state": state, "why": why,
                           "net_px": None if r is None else r["net_px"],
                           "ncc_median": None if r is None else r["ncc_median"]})

    total = sum(counts.values())
    print(f"appearance-based audit of {total} takes in {LAB.name}")
    print(f"  motion counted above {MOVE_WIDTHS:.2f} object-widths END TO END; "
          f"appearance intact above ncc {NCC_OK:.2f}\n")
    states = ["MOVES", "STATIC", "DEGRADED", "NOTRACK"]
    hdr = f"{'probe':10s}" + "".join(f"{s:>11s}" for s in states)
    print(hdr); print("-" * len(hdr))
    for kind, d in sorted(per_kind.items()):
        print(f"{kind:10s}" + "".join(f"{d.get(s,0):>11d}" for s in states))
    print("-" * len(hdr))
    print(f"{'ALL':10s}" + "".join(f"{counts.get(s,0):>11d}" for s in states))
    for s in states:
        n = counts.get(s, 0)
        print(f"  {s:9s} {n:3d}/{total}  ({100.0*n/max(total,1):.0f}%)")

    print("\nDEGRADED takes — the object stopped being the object")
    deg = [d for d in detail if d["state"] == "DEGRADED"]
    for d in deg[:14]:
        print(f"  {d['clip']:38s} {d['why']}")
    if len(deg) > 14:
        print(f"  ... and {len(deg)-14} more")

    (LAB / "audit_pixel.json").write_text(json.dumps(
        {"counts": counts, "per_kind": per_kind, "detail": detail},
        indent=2, default=float))
    print(f"\nwrote {LAB}/audit_pixel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
