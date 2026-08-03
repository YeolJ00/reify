"""Animations of the generated clips themselves, for the report.

The report has shown measurements and simulations of these clips without ever showing the
clips. The three failure modes -- an object that never moves, an object that stops being
itself, and a clip that actually works -- are far more legible as animation than as a yield
percentage, and the prompt A/B is a genuine before/after on the same staged frame.

Run: python scripts/make_clip_figs.py            (warp env, no GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXP = REPO / "outputs" / "scene" / "expand"
AB = REPO / "outputs" / "scene" / "ab"
DOCS = REPO / "docs"
STEP, SCALE = 2, 0.62


def font(sz):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return None


def strip(panes, labels, colours):
    """panes: list of frame arrays, laid out side by side with a label bar."""
    f = font(12)
    n = min(len(p) for p in panes)
    out = []
    for t in range(0, n, STEP):
        ims = [Image.fromarray(p[t]) for p in panes]
        w, h = ims[0].size
        nw, nh = int(w * SCALE), int(h * SCALE)
        ims = [im.resize((nw, nh), Image.LANCZOS) for im in ims]
        bar = 20
        c = Image.new("RGB", (nw * len(ims) + 6 * (len(ims) - 1), nh + bar),
                      (245, 247, 250))
        d = ImageDraw.Draw(c)
        for i, im in enumerate(ims):
            x = i * (nw + 6)
            c.paste(im, (x, bar))
            d.rectangle([x, 0, x + nw, bar - 1], fill=colours[i])
            d.text((x + 6, 4), labels[i], fill=(240, 244, 248), font=f)
        out.append(c)
    return out


def save(frames, name):
    p = DOCS / f"{name}.webp"
    frames[0].save(p, save_all=True, append_images=frames[1:],
                   duration=int(1000 / 24 * STEP), loop=0, quality=68, method=6)
    return p, p.stat().st_size / 1024


def load(p):
    return np.load(p)["frames"]


def main():
    made = {}
    DARK, TEAL, RED = (20, 26, 34), (14, 124, 147), (150, 40, 34)

    # 1. the prompt fix, same staged frame, same seed
    a = AB / "vid_brass_pot_control_seed1.npz"
    b = AB / "vid_brass_pot_neg_seed2.npz"
    if a.exists() and b.exists():
        fr = strip([load(a), load(b)],
                   ["NO NEGATIVE PROMPT — becomes a bowl", "WITH NEGATIVE PROMPT — stays a pot"],
                   [RED, TEAL])
        p, kb = save(fr, "clip_prompt_ab")
        made["clip_prompt_ab"] = {"kb": round(kb, 1),
                                  "caption": "Brass pot, dropped. Identical staged frame; "
                                             "the only difference is the negative prompt."}
        print(f"  clip_prompt_ab       {kb:6.1f} KB")

    # 2. the three outcomes a generated clip can have
    trio = [("ceramic_vase_drop_mid_seed5", "USABLE — falls, lands, stays itself", TEAL),
            ("rubber_duck_drop_low_seed3", "NO MOTION — nothing happens", DARK),
            ("brass_pot_drop_mid_seed2", "DEGRADED — stops being the object", RED)]
    panes, labels, cols = [], [], []
    for stem, lab, col in trio:
        p = EXP / f"vid_{stem}.npz"
        if p.exists():
            panes.append(load(p)); labels.append(lab); cols.append(col)
    if len(panes) >= 2:
        fr = strip(panes, labels, cols)
        p, kb = save(fr, "clip_outcomes")
        made["clip_outcomes"] = {"kb": round(kb, 1),
                                 "caption": "The three things a generated clip can do. "
                                            "Only the first is a measurement."}
        print(f"  clip_outcomes        {kb:6.1f} KB")

    (DOCS / "clips.json").write_text(json.dumps(made, indent=2))
    print(f"\n{len(made)} clip animations, "
          f"{sum(m['kb'] for m in made.values()):.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
