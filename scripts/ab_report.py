"""Score the prompting A/B with the instruments we already trust.

Not by eye. Each clip is judged on the two things that have actually been blocking us:

    asset intact   NCC of the object's own patch against frame 0 stays above 0.55
                   (the brass pot turning into a bowl fails here)
    measurable     a restitution can be extracted at all, and lies in [0, 1]

Yield is the fraction of clips clearing both. The arms differ only in the prompt, and the
staged frame is byte-identical across them, so any difference is the prompting.

Run: python scripts/ab_report.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.motion.observables import admissible, drop_observables  # noqa: E402
from src.motion.patch_track import track_patch  # noqa: E402

OUT = REPO / "outputs" / "scene" / "ab"
EXP = REPO / "outputs" / "scene" / "expand"
NCC_OK = 0.55
ARM_ORDER = ["control", "neg", "json", "json_neg"]
LABEL = {"control": "stub prompt (what we shipped)",
         "neg": "stub + NVIDIA negative prompt",
         "json": "documented JSON structure",
         "json_neg": "JSON structure + negative prompt"}


def main():
    seeds = json.loads((EXP / "seeds_image.json").read_text())
    rows = {}
    for f in sorted(OUT.glob("vid_*.npz")):
        stem = f.stem.replace("vid_", "")
        parts = stem.rsplit("_seed", 1)
        body = parts[0]
        # longest arm name first: "json_neg" also ends with "_neg", and matching the
        # short one made the subject parse as "<obj>_json", which has no seed entry, so
        # all 12 json_neg clips were silently skipped rather than mis-binned
        arm = next((a for a in sorted(ARM_ORDER, key=len, reverse=True)
                    if body.endswith("_" + a)), None)
        if arm is None:
            print(f"  ! unrecognised arm in {stem}")
            continue
        subj = body[: -len(arm) - 1]
        key = f"{subj}_drop_mid"
        s = seeds.get(key, {}).get("subject")
        if not s:
            print(f"  ! no seed for {key} (from {stem}) — clip dropped")
            continue
        cache = OUT / f"ptrk_{stem}.npz"
        if cache.exists():
            d = np.load(cache)
            r = {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}
        else:
            try:
                fr = np.load(f)["frames"]
            except Exception:
                continue
            r = track_patch(fr, s["u"], s["v"], half=max(s["size_px"] * 0.45, 10),
                            search=140)
            if r is None:
                continue
            np.savez_compressed(cache, **{k: np.asarray(v) for k, v in r.items()})
        intact = r["ncc_median"] >= NCC_OK and r["ncc_end"] >= NCC_OK
        o = drop_observables(np.stack([r["u"], r["v"]], 1))
        ok = bool(intact and o.get("ok") and admissible("restitution", o["value"]))
        rows.setdefault((subj, arm), []).append(
            {"intact": bool(intact), "ok": ok,
             "e": float(o["value"]) if o.get("ok") else None,
             "ncc": float(r["ncc_median"])})

    subjects = sorted({k[0] for k in rows})
    print(f"{'arm':34s}" + "".join(f"{s:>22s}" for s in subjects) + f"{'ALL':>14s}")
    print("-" * (34 + 22 * len(subjects) + 14))
    tot = {}
    for arm in ARM_ORDER:
        cells, g, n, ig, deg = "", 0, 0, 0, 0
        for s in subjects:
            v = rows.get((s, arm), [])
            if not v:
                cells += f"{'—':>22s}"; continue
            a = sum(x["ok"] for x in v); b = len(v)
            di = sum(1 for x in v if not x["intact"])
            g += a; n += b; deg += di
            cells += f"{f'{a}/{b} ok, {di} degraded':>22s}"
        tot[arm] = (g, n, deg)
        allc = f"{g}/{n}" if n else "—"
        print(f"{LABEL[arm]:34s}{cells}{allc:>14s}")
    print("-" * (34 + 22 * len(subjects) + 14))
    print("\nyield (usable restitution) and asset degradation, pooled:")
    base = tot.get("control", (0, 1, 0))
    for arm in ARM_ORDER:
        g, n, deg = tot[arm]
        if not n:
            continue
        dy = (100.0 * g / n) - (100.0 * base[0] / max(base[1], 1))
        print(f"  {LABEL[arm]:34s} yield {100.0*g/n:5.1f}%  "
              f"({dy:+5.1f} pts vs control)   degraded {100.0*deg/n:5.1f}%")
    (OUT / "ab_report.json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, indent=2, default=float))
    print(f"\nwrote {OUT}/ab_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
