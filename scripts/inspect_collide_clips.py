"""Look at the collision clips before drawing any conclusion from their fits.

Two takes of the same vase collision fit confidently to mass ratios 15.8x apart. That is
either (a) the generator produced two physically different events, or (b) the tracker
followed the wrong thing in one of them. We have made mistake (b) before and called it a
physics finding, so the clips get inspected directly: a filmstrip per take with the
tracked subject (cyan) and partner (magenta) centroids drawn on the actual frames, plus
the observed signature numbers underneath.

Run: python scripts/inspect_collide_clips.py <subject> [seeds ...]     (warp env)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.motion.observables import collide_observables  # noqa: E402

LAB = REPO / "outputs" / "scene" / "fulllab"
NCOL = 8


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "ceramic_vase"
    seeds = sys.argv[2:] or ["0", "2"]

    rows = []
    for s in seeds:
        v = LAB / f"vid_{subj}_collide_seed{s}.npz"
        t = LAB / f"trk_{subj}_collide_seed{s}.npz"
        if not (v.exists() and t.exists()):
            print(f"missing seed {s}"); continue
        fr = np.load(v)["frames"]
        d = np.load(t)
        rows.append((s, fr, d["subject_cen"], d["partner_cen"]))

    if not rows:
        return 1

    nf = rows[0][1].shape[0]
    idx = np.linspace(0, nf - 1, NCOL).round().astype(int)
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(2.05 * NCOL, 2.0 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (s, fr, sc, pc) in enumerate(rows):
        sg = collide_observables(sc, pc)
        for c, fi in enumerate(idx):
            ax = axes[r, c]; ax.imshow(fr[fi]); ax.set_xticks([]); ax.set_yticks([])
            if np.isfinite(sc[fi]).all():
                ax.plot(*sc[fi], "o", mfc="none", mec="#22d3ee", ms=13, mew=2.2)
            if np.isfinite(pc[fi]).all():
                ax.plot(*pc[fi], "o", mfc="none", mec="#f0f", ms=13, mew=2.2)
            ax.set_title(f"f{fi}", fontsize=8, pad=2)
        lab = f"seed {s}\n"
        if not sg.get("ok"):
            lab += "NO MEASUREMENT\n" + str(sg.get("why", ""))[:38]
        else:
            lab += (f"m_t/m_m={sg['value']:.2f}+-{sg['se']:.2f}\n"
                    f"v_pre={sg['v_pre']:.2f}\nv_post={sg['v_post']:.2f}\n"
                    f"v_target={sg['v_t']:.2f}")
        axes[r, 0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
        # how far each tracked point moved in total, a blunt check that it tracked anything
        for nm, arr in (("subject", sc), ("partner", pc)):
            fin = arr[np.isfinite(arr[:, 0])]
            tot = float(np.abs(np.diff(fin[:, 0])).sum()) if len(fin) > 1 else 0.0
            span = (float(fin[:, 0].min()), float(fin[:, 0].max())) if len(fin) else (0, 0)
            print(f"  seed {s} {nm:8s}: x travel {tot:6.1f}px  x range {span[0]:5.0f}..{span[1]:5.0f}"
                  f"  nan {np.isnan(arr[:,0]).mean()*100:.0f}%")
        msg = (sg.get("why") if not sg.get("ok")
               else f"m_t/m_m={sg['value']:.3f}+-{sg['se']:.3f}")
        print(f"  seed {s}: {msg}")

    fig.suptitle(f"{subj} collide — cyan=subject  magenta=partner", fontsize=10)
    fig.tight_layout(rect=[0.045, 0, 1, 0.96])
    out = LAB / f"inspect_{subj}_collide.png"
    fig.savefig(out, dpi=105); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
