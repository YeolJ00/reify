"""Render the generated-vs-simulated comparisons as animated GIFs and filmstrips.

Both panes are cropped to the same window -- the union of where the object goes in the
video and where it goes in the simulation -- so the two are directly comparable and the
file stays small enough to embed in a self-contained page.

Run: python scripts/make_comparison_figs.py         (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"
DOCS = REPO / "docs"
PAD = 26
MAXW = 240
STEP = 2          # 24fps -> 12fps: halves the payload, motion still reads clearly

# GIF of these at full rate came to 4.3 MB EACH, which cannot go in a self-contained
# page. Animated WebP carries the same 49-frame comparison at a fraction of the size.


def motion_window(gen, sim):
    """Bounding box of everything that changes in either sequence, plus padding."""
    H, W = gen.shape[1], gen.shape[2]
    box = None
    for seq in (gen, sim):
        d = np.abs(seq.astype(np.int16) - seq[0].astype(np.int16)).mean(3).max(0)
        ys, xs = np.where(d > 16)
        if len(xs) < 20:
            continue
        b = [xs.min(), ys.min(), xs.max(), ys.max()]
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3])]
    if box is None:
        box = [0, 0, W - 1, H - 1]
    x0 = max(int(box[0]) - PAD, 0); y0 = max(int(box[1]) - PAD, 0)
    x1 = min(int(box[2]) + PAD, W - 1); y1 = min(int(box[3]) + PAD, H - 1)
    if x1 - x0 < 60:
        x0, x1 = max(x0 - 40, 0), min(x1 + 40, W - 1)
    if y1 - y0 < 60:
        y0, y1 = max(y0 - 40, 0), min(y1 + 40, H - 1)
    return x0, y0, x1, y1


def stack(gen, sim, box, scale):
    x0, y0, x1, y1 = box
    out = []
    for t in range(len(gen)):
        a = Image.fromarray(gen[t][y0:y1, x0:x1])
        b = Image.fromarray(sim[t][y0:y1, x0:x1])
        w, h = a.size
        nw, nh = int(w * scale), int(h * scale)
        a = a.resize((nw, nh), Image.LANCZOS); b = b.resize((nw, nh), Image.LANCZOS)
        # a label strip: without it the two panes are indistinguishable, which is the
        # one thing this figure must never be ambiguous about
        bar = 15
        c = Image.new("RGB", (nw * 2 + 6, nh + bar), (245, 247, 250))
        c.paste(a, (0, bar)); c.paste(b, (nw + 6, bar))
        dr = ImageDraw.Draw(c)
        dr.rectangle([0, 0, nw, bar - 1], fill=(20, 26, 34))
        dr.rectangle([nw + 6, 0, nw * 2 + 6, bar - 1], fill=(14, 124, 147))
        dr.text((5, 3), "GENERATED (Cosmos)", fill=(235, 240, 245))
        dr.text((nw + 11, 3), "SIMULATED (Newton)", fill=(235, 245, 248))
        out.append(c)
    return out


def main():
    meta = json.loads((LAB / "sim_videos.json").read_text())
    made = {}
    for key, info in meta.items():
        p = LAB / f"cmp_{key}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        gen, sim = d["generated"], d["simulated"]
        box = motion_window(gen, sim)
        scale = min(1.0, MAXW / max(box[2] - box[0], 1))
        frames = stack(gen[::STEP], sim[::STEP], box, scale)
        anim = DOCS / f"cmp_{key}.webp"
        frames[0].save(anim, save_all=True, append_images=frames[1:],
                       duration=int(1000 / 24 * STEP), loop=0, quality=62, method=6)
        kb = anim.stat().st_size / 1024
        # a static filmstrip too: the report should still read with animation blocked
        idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
        w, h = frames[0].size
        strip = Image.new("RGB", (w * len(idx) + 5 * (len(idx) - 1), h), (245, 247, 250))
        for i, fi in enumerate(idx):
            strip.paste(frames[fi], (i * (w + 5), 0))
        strip.save(DOCS / f"cmp_{key}.jpg", quality=84, optimize=True)
        made[key] = {"anim_kb": round(kb, 1), "file": anim.name, "frames": len(frames),
                     "box": [int(x) for x in box], "note": info["note"]}
        print(f"  {key:26s} webp {kb:6.1f} KB  {len(frames)} frames  {info['note']}")
    (DOCS / "comparisons.json").write_text(json.dumps(made, indent=2))
    tot = sum(m["anim_kb"] for m in made.values())
    print(f"\n{len(made)} comparisons, {tot:.0f} KB of animation total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
