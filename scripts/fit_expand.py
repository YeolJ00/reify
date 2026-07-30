"""Fit the expanded lab: restitution at three impact speeds, plus friction.

Three drop heights turn restitution from one number into a relationship. For most real
materials e falls as impact speed rises, so the pair

    e_mid  restitution at the CENTRE of the measured speed range
    de/dv  how fast it falls with speed   (1/(m/s))

The fit is centred on the mean impact speed deliberately. Reporting e at v=0 sounded
tidier but is an extrapolation outside the data -- these clips span roughly 1.25 to
2.4 m/s -- and on a partial run it produced e0 = -0.481 for the baseball, a negative
restitution, from a perfectly ordinary set of takes. Centring also decorrelates the
intercept from the slope, so their error bars mean what they appear to mean.

is a strictly richer description than a single e, and it comes from the highest-yielding
probe we have rather than from a new one that might not yield at all.

The impact speed is not assumed from the staged height -- the video model does not
promise to obey g -- it is MEASURED from the same track, as the fitted approach speed
converted through the pixel scale. A height that produced no visible fall therefore
contributes nothing rather than contributing a wrong x-value.

Run: python scripts/fit_expand.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.simple_fit import pixel_scale  # noqa: E402
from src.motion import observables as OB  # noqa: E402
from src.motion.observables import (admissible, combine, drop_observables,  # noqa: E402
                                    polyfit_se, slide_observables)
from src.motion.patch_track import track_patch  # noqa: E402

LAB = REPO / "outputs" / "scene" / "expand"
FPS = 24.0
NCC_OK = 0.55
SEARCH = 140


def track_cached(f, seed, role):
    """Track with a disk cache, tolerating clips the generator is still writing.

    This script is meant to be runnable while generation is in flight, so a half-written
    npz must be skipped rather than crash the run -- and a cache entry must never be
    written from a partial read.
    """
    cache = LAB / f"ptrk_{f.stem.replace('vid_','')}_{role}.npz"
    if cache.exists():
        try:
            d = np.load(cache)
            return {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}
        except Exception:
            cache.unlink(missing_ok=True)          # truncated cache entry
    try:
        fr = np.load(f)["frames"]
    except Exception:
        return "INCOMPLETE"
    r = track_patch(fr, seed["u"], seed["v"],
                    half=max(seed["size_px"] * 0.45, 10), search=SEARCH)
    if r is not None:
        np.savez_compressed(cache, **{k: np.asarray(v) for k, v in r.items()})
    return r


def main():
    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    px_per_m, _ = pixel_scale(cfg["camera"], cfg["ground_z"])

    per = {}          # subj -> {"drop": [(v_impact, e, se)], "slide": ([v],[se])}
    noise, nclip, ndeg, nskip = [], 0, 0, 0
    for key, e in sorted(cfg["experiments"].items()):
        s = seeds.get(key, {}).get("subject")
        if not s:
            continue
        for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
            r = track_cached(f, s, "subject")
            if r == "INCOMPLETE":
                nskip += 1
                continue
            nclip += 1
            if r is None:
                continue
            if r["ncc_median"] < NCC_OK or r["ncc_end"] < NCC_OK:
                ndeg += 1
                continue
            cen = np.stack([r["u"], r["v"]], 1)
            b = per.setdefault(e["subject"], {"drop": [], "slide_v": [], "slide_s": []})
            if e["kind"] == "drop":
                o = drop_observables(cen)
                if not o.get("ok") or not admissible("restitution", o["value"]):
                    continue
                # measured impact speed in m/s, not the staged height
                v_imp = abs(o["v_pre"]) * FPS / px_per_m
                b["drop"].append((v_imp, o["value"], o["se"], e["height_name"]))
            else:
                o = slide_observables(cen, px_per_m, FPS)
                if not o.get("ok") or not admissible("friction", o["value"]):
                    continue
                b["slide_v"].append(o["value"]); b["slide_s"].append(o["se"])
                if o["value"] < 1e-3:
                    noise.append(o["se"])

    print(f"{nclip} clips read, {ndeg} dropped as degraded"
          + (f", {nskip} still being written" if nskip else "") + "\n")
    print(f"{'object':14s} {'n':>3s} {'e at mid speed':>18s} {'de/dv':>18s} "
          f"{'friction mu':>18s}    v_mid")
    print("-" * 74)
    res = {}
    for subj in sorted(per):
        b = per[subj]
        d = b["drop"]
        row = {"n_drop": len(d)}
        if len(d) >= 3 and len({x[3] for x in d}) >= 2:
            v = np.array([x[0] for x in d]); ee = np.array([x[1] for x in d])
            se = np.array([x[2] for x in d])
            # weighted straight line e = e0 + slope*v, weights from each take's own error
            w = 1.0 / np.maximum(se, 1e-6) ** 2
            vbar = float(np.average(v, weights=w))
            V = np.stack([np.ones_like(v), v - vbar], 1)
            A = V.T @ (w[:, None] * V)
            try:
                cov = np.linalg.inv(A)
                beta = cov @ (V.T @ (w * ee))
                e_mid, slope = float(beta[0]), float(beta[1])
                se0, sslope = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
                row.update(e_mid=e_mid, e_mid_se=se0, v_mid=vbar,
                           slope=slope, slope_se=sslope,
                           physical=bool(0.0 <= e_mid <= 1.0),
                           v_range=[float(v.min()), float(v.max())])
            except np.linalg.LinAlgError:
                pass
        c = combine(b["slide_v"], b["slide_s"]) if b["slide_v"] else None
        row["friction"] = c
        res[subj] = row
        es = ("—" if "e_mid" not in row else
              f"{row['e_mid']:.3f} ± {row['e_mid_se']:.3f}"
              + ("" if row.get("physical", True) else " !"))
        sl = ("—" if "slope" not in row
              else f"{row['slope']:+.3f} ± {row['slope_se']:.3f}")
        fr = ("—" if not c else f"{c['value']:.3f} ± {c['interval']:.3f} (n={c['n']})")
        vm = ("" if "v_mid" not in row else f"  {row['v_mid']:.2f} m/s")
        print(f"{subj:14s} {row['n_drop']:>3d} {es:>18s} {sl:>18s} {fr:>18s}{vm}")

    print("\nde/dv is the change in restitution per m/s of impact speed; a value whose")
    print("interval spans zero means these clips do not resolve any speed dependence.")
    sig = [s for s, r in res.items()
           if "slope" in r and abs(r["slope"]) > 2 * r["slope_se"]]
    print(f"resolved for {len(sig)} of {len(res)} objects"
          + (f": {', '.join(sig)}" if sig else ""))
    print("\nparameters measured to better than +-25% from more than one take:")
    tight = []
    for subj, r in res.items():
        if "e_mid" in r and r["n_drop"] > 1 and r["e_mid"] > 0 \
                and r["e_mid_se"] / r["e_mid"] < 0.25:
            tight.append((f"{subj} restitution", r["e_mid"], r["e_mid_se"]))
        c = r.get("friction")
        if c and c["n"] > 1 and c["value"] > 0 and c["interval"] / c["value"] < 0.25:
            tight.append((f"{subj} friction", c["value"], c["interval"]))
    for nm, v, e in tight:
        print(f"  {nm:26s} {v:.3f} ± {e:.3f}   ({100*e/v:.0f}%)")
    if not tight:
        print("  none")
    print(f"  -> {len(tight)} parameters, against 0 before the expansion")

    bad = [s for s, r in res.items() if r.get("physical") is False]
    if bad:
        print(f"! e outside [0,1] for {', '.join(bad)} — not a measurement")

    (LAB / "expand_fit.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {LAB}/expand_fit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
