"""Locate each subject in its own rendered frame, instead of projecting geometry at it.

seeds.json was built by projecting `obj_geom`'s body centre through the camera. That
centre is `pos + vmean` with no rotation applied, which does not match how Blender places
the object (origin at `pos`, rotated by rot_z), so the seed landed off the object by more
than the tracking patch's half-size for four of seven objects:

    ceramic vase  +35 px   (patch half-size 28)
    rubber duck   +36 px   (29)
    book          +27 px   (16)
    brass pot     +40 px   (41, and -88 in v)

Every downstream measurement was therefore tracking a patch that was substantially wall.

The fix is not to correct the projection but to remove the parallel pipeline: the subject
is the only thing that moves between two stagings of the same object, so differencing two
initial frames isolates it exactly, in the image we are going to track. Geometry
conventions cannot disagree with the renderer if the renderer is the source.

Run: LAB=<dir> python scripts/seed_from_image.py        (warp env, no GPU)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = Path(os.environ.get("LAB") or (REPO / "outputs" / "scene" / "expand"))
DIFF = 30
MIN_AREA = 250


def blobs(a, b):
    d = (np.abs(a - b).mean(2) > DIFF).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(d, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        out.append({"u": float(cent[i][0]), "v": float(cent[i][1]),
                    "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                    "area": int(stats[i, cv2.CC_STAT_AREA])})
    return out


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    old = json.loads((LAB / "seeds.json").read_text()) if (LAB / "seeds.json").exists() else {}
    imgs = {k: np.asarray(Image.open(LAB / f"I0_{k}.png").convert("RGB"), float)
            for k in cfg["experiments"] if (LAB / f"I0_{k}.png").exists()}

    by_subj = {}
    for k, e in cfg["experiments"].items():
        by_subj.setdefault(e["subject"], []).append(k)

    out, moved = {}, []
    for subj, keys in by_subj.items():
        ref = next((k for k in keys if cfg["experiments"][k]["kind"] == "slide"), None)
        if ref is None or ref not in imgs:
            print(f"  {subj}: no slide reference"); continue
        # the resting position, from any lifted staging: the LOWER of the two blobs
        anchor = None
        for k in keys:
            if k == ref or k not in imgs:
                continue
            bs = blobs(imgs[k], imgs[ref])
            if len(bs) >= 2:
                anchor = max(bs, key=lambda b: b["v"])
                break
        for k in keys:
            if k not in imgs:
                continue
            if k == ref:
                b = anchor
            else:
                bs = blobs(imgs[k], imgs[ref])
                if not bs:
                    continue
                # the subject in THIS staging is the blob that is not the resting one
                b = (min(bs, key=lambda x: x["v"]) if anchor is None else
                     max(bs, key=lambda x: (x["v"] - anchor["v"]) ** 2
                         + (x["u"] - anchor["u"]) ** 2))
            if b is None:
                continue
            size_px = float(min(b["w"], b["h"]))
            s = {"u": b["u"], "v": b["v"], "r": float(0.40 * size_px),
                 "size_px": size_px, "w_px": float(b["w"]), "h_px": float(b["h"]),
                 "source": "image"}
            out[k] = {"subject": s}
            o = (old.get(k) or {}).get("subject")
            if o:
                moved.append((k, o["u"] - s["u"], o["v"] - s["v"]))

    (LAB / "seeds_image.json").write_text(json.dumps(out, indent=2))
    print(f"located {len(out)} subjects from the rendered frames")
    if moved:
        a = np.array([[m[1], m[2]] for m in moved], float)
        big = [m for m in moved if abs(m[1]) > 20 or abs(m[2]) > 20]
        print(f"median shift from the old projected seeds: "
              f"u {np.median(a[:,0]):+.0f} px, v {np.median(a[:,1]):+.0f} px")
        print(f"{len(big)} of {len(moved)} seeds move by more than 20 px:")
        for k, du, dv in sorted(big, key=lambda m: -abs(m[1]))[:10]:
            print(f"    {k:28s} {du:+6.0f},{dv:+5.0f}")
    print(f"\nwrote {LAB}/seeds_image.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
