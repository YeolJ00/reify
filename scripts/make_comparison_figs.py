"""Generated clip beside Newton's own render of the simulation. No masks, no compositing.

Left: what Cosmos produced. Right: newton.viewer.ViewerRTX rendering the simulated mesh at
the pose our rollout computed -- ray traced, real geometry, real pose. ProbeScene is pure
Warp and never builds a newton.Model, which is why Newton's viewers had nothing to draw;
the geometry is loaded into a ModelBuilder purely for display and driven from our rollout.

The two panes differ in surroundings (Newton draws its own ground and lighting, not the
wooden table and HDRI the clip was staged in) and in framing, since ViewerRTX exposes
pitch/yaw but not the lab camera's 46 degree field of view. Only the motion is comparable.

Run: python scripts/make_comparison_figs.py         (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"
DOCS = REPO / "docs"
STEP = 2
SCALE = 0.86


def font(sz):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return None


def stack(gen, sim):
    out = []
    f = font(13)
    for t in range(min(len(gen), len(sim))):
        a = Image.fromarray(gen[t]); b = Image.fromarray(sim[t])
        w, h = a.size
        nw, nh = int(w * SCALE), int(h * SCALE)
        a = a.resize((nw, nh), Image.LANCZOS); b = b.resize((nw, nh), Image.LANCZOS)
        bar = 22
        c = Image.new("RGB", (nw * 2 + 6, nh + bar), (245, 247, 250))
        c.paste(a, (0, bar)); c.paste(b, (nw + 6, bar))
        d = ImageDraw.Draw(c)
        d.rectangle([0, 0, nw, bar - 1], fill=(20, 26, 34))
        d.rectangle([nw + 6, 0, nw * 2 + 6, bar - 1], fill=(14, 124, 147))
        d.text((7, 5), "GENERATED (Cosmos)", fill=(235, 240, 245), font=f)
        d.text((nw + 13, 5), "SIMULATED (Newton ViewerRTX)", fill=(235, 245, 248), font=f)
        out.append(c)
    return out


def main():
    meta = json.loads((LAB / "newton_render.json").read_text())
    cfg = json.loads((LAB / "lab.json").read_text())
    made = {}
    for key, info in meta.items():
        sdir = LAB / info["dir"]
        fs = sorted(sdir.glob("f*.png"))
        if not fs:
            continue
        sim = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fs])
        # the generated take this parameter came from
        cands = sorted(LAB.glob(f"vid_{key}_seed*.npz"))
        pick = None
        for f in cands:
            if (LAB / f"ptrk_img_{f.stem.replace('vid_','')}_subject.npz").exists():
                pick = f
                break
        if pick is None:
            continue
        gen = np.load(pick)["frames"]
        frames = stack(gen[::STEP], sim[::STEP])
        anim = DOCS / f"cmp_{key}.webp"
        frames[0].save(anim, save_all=True, append_images=frames[1:],
                       duration=int(1000 / 24 * STEP), loop=0, quality=70, method=6)
        idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
        w, h = frames[0].size
        strip = Image.new("RGB", (w * len(idx) + 5 * (len(idx) - 1), h), (245, 247, 250))
        for i, fi in enumerate(idx):
            strip.paste(frames[fi], (i * (w + 5), 0))
        strip.save(DOCS / f"cmp_{key}.jpg", quality=84, optimize=True)
        kb = anim.stat().st_size / 1024
        made[key] = {"anim_kb": round(kb, 1), "file": anim.name,
                     "frames": len(frames), "note": info["note"]}
        print(f"  {key:26s} webp {kb:6.1f} KB  {len(frames)} frames  {info['note']}")
    (DOCS / "comparisons.json").write_text(json.dumps(made, indent=2))
    print(f"\n{len(made)} comparisons, "
          f"{sum(m['anim_kb'] for m in made.values()):.0f} KB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
