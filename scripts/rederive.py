"""Re-derive every parameter from appearance tracks instead of point tracks.

The published values came from CoTracker centroids, and the M40 audit found that layer
unreliable: points stay "visible" while sitting on the background, so a duck that visibly
fell measured 6 px of motion. This recomputes the same three parameters from NCC patch
tracking (src/motion/patch_track.py), which reports where the object matches AND how well,
and prints the two side by side.

Two things this does that the point-track version could not:

  * DEGRADED takes are dropped. If the object stops being the object -- our brass pot
    turns into a bowl mid-clip -- then no rigid-body theta explains it at all, and
    including it would be fitting physics to a rendering artefact.
  * The tracker noise floor is MEASURED rather than assumed, from the residual scatter
    of takes in which nothing moves. Those clips contain only noise by construction, so
    they calibrate the error bars on every other clip.

Run: python scripts/rederive.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.simple_fit import pixel_scale  # noqa: E402
from src.motion import observables as OB  # noqa: E402
from src.motion.observables import (admissible, collide_observables,  # noqa: E402
                                    combine, drop_observables, slide_observables)
from src.motion.patch_track import track_patch  # noqa: E402

LAB = REPO / "outputs" / "scene" / "fulllab"
FPS = 24.0
NCC_OK = 0.55
MOVE_WIDTHS = 0.30
SEARCH = 120
PROBE = {"drop": ("restitution", "e", "restitution"),
         "slide": ("friction", "mu", "friction"),
         "collide": ("mass ratio", "m_t/m_m", "mass_ratio")}


def track_one(frames, seed):
    return track_patch(frames, seed["u"], seed["v"],
                       half=max(seed["size_px"] * 0.45, 10), search=SEARCH)


def cen_of(r):
    return np.stack([r["u"], r["v"]], 1)


def residual_scatter(r):
    """Frame-to-frame scatter after removing a linear trend -- i.e. the noise."""
    c = cen_of(r)
    m = np.isfinite(c[:, 0])
    if m.sum() < 8:
        return np.nan
    t = np.arange(len(c))[m]
    s = []
    for k in (0, 1):
        y = c[m, k]
        b = np.polyfit(t, y, 1)
        s.append(np.std(y - np.polyval(b, t)))
    return float(np.hypot(*s) / np.sqrt(2))


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    old = json.loads((LAB / "simple_fit.json").read_text())
    px_per_m, _ = pixel_scale(cfg["camera"], cfg["ground_z"])

    # ---- pass 1: track everything, classify, and collect the noise calibration
    tracks, noise = {}, []
    for key, e in sorted(cfg["experiments"].items()):
        s = seeds.get(key, {}).get("subject")
        if not s:
            continue
        for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
            fr = np.load(f)["frames"]
            r = track_one(fr, s)
            if r is None:
                continue
            p = None
            ps = seeds[key].get("partner")
            if e["kind"] == "collide" and ps is not None:
                p = track_one(fr, ps)
            degraded = (r["ncc_median"] < NCC_OK or r["ncc_end"] < NCC_OK)
            moved = (r["end_px"] / max(s["size_px"], 1e-6)) >= MOVE_WIDTHS
            if not degraded and not moved:
                sc = residual_scatter(r)
                if np.isfinite(sc):
                    noise.append(sc)
            tracks[f.stem.replace("vid_", "")] = {
                "key": key, "kind": e["kind"], "subj": e["subject"], "r": r, "p": p,
                "size_px": s["size_px"], "degraded": degraded, "moved": moved}

    floor = float(np.median(noise)) if noise else OB.SIGMA_FLOOR_PX
    print(f"tracker noise floor measured from {len(noise)} static takes: "
          f"{floor:.2f} px  (was assuming {OB.SIGMA_FLOOR_PX:.2f})")
    OB.SIGMA_FLOOR_PX = max(floor, 0.3)
    print(f"using {OB.SIGMA_FLOOR_PX:.2f} px\n")

    # ---- pass 2: parameters
    out, dropped = {}, {"degraded": 0, "static": 0, "no_obs": 0, "contra": 0}
    for name, t in sorted(tracks.items()):
        subj, kind = t["subj"], t["kind"]
        bucket = out.setdefault(subj, {}).setdefault(kind, {"v": [], "s": []})
        if t["degraded"]:
            dropped["degraded"] += 1
            continue
        cen = cen_of(t["r"])
        if kind == "drop":
            o = drop_observables(cen)
        elif kind == "slide":
            o = slide_observables(cen, px_per_m, FPS)
        else:
            o = ({"ok": False} if t["p"] is None
                 else collide_observables(cen, cen_of(t["p"])))
        if not o.get("ok"):
            dropped["no_obs"] += 1
            continue
        if not admissible(PROBE[kind][2], o["value"]):
            dropped["contra"] += 1
            continue
        bucket["v"].append(o["value"]); bucket["s"].append(o["se"])

    print(f"takes dropped: {dropped['degraded']} degraded, "
          f"{dropped['no_obs']} no observable, {dropped['contra']} contradicted physics\n")

    res = {}
    for subj, per in out.items():
        res[subj] = {}
        for kind in ("drop", "slide", "collide"):
            b = per.get(kind)
            c = combine(b["v"], b["s"]) if b else None
            res[subj][kind] = c

    print(f"{'object':14s} {'probe':8s} {'point tracks (published)':>28s} "
          f"{'appearance tracks (new)':>28s}")
    print("-" * 84)
    for subj in sorted(res):
        for kind in ("drop", "slide", "collide"):
            n = res[subj].get(kind)
            o = (old.get(subj) or {}).get(kind) or {}
            os_ = ("—" if o.get("value") is None
                   else f"{o['value']:.3f} ± {o['interval']:.3f} (n={o.get('n','?')})")
            ns = ("—" if not n else f"{n['value']:.3f} ± {n['interval']:.3f} "
                                    f"(n={n['n']})")
            print(f"{subj:14s} {kind:8s} {os_:>28s} {ns:>28s}")

    tight = sum(1 for s in res.values() for c in s.values()
                if c and not c["single_take"]
                and c["interval"] / max(abs(c["value"]), 1e-9) < 0.25)
    got = sum(1 for s in res.values() for c in s.values() if c)
    print(f"\n{got} of 15 parameters produced a value; {tight} of those are tighter "
          f"than ±25% from more than one take")

    (LAB / "rederived.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {LAB}/rederived.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
