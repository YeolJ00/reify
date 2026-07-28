"""Figure: held-out prediction on generated video.

Material recovered jointly from the drop + collide videos is frozen, then used to
predict a FOURTH generated video (a differently-configured collision) that was never
fitted — only its unknown launch velocity is fitted, because an initial condition
cannot transfer but a material must.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.render.camera import Camera  # noqa: E402

OUT = REPO / "outputs" / "probes_i2v"
BG = "#141210"


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(OUT / "joint_heldout.npz", allow_pickle=True)
    cfg = json.loads((OUT / "scene.json").read_text())
    cam = Camera(cfg["camera"])
    th = json.loads(str(d["theta"])); dflt = json.loads(str(d["default"]))
    Ah, Bh, Pj, Pd = d["Ah"], d["Bh"], d["Pj"], d["Pd"]
    r_j, r_d = float(d["r_joint"]), float(d["r_def"])
    frames = np.load(OUT / "vid_heldout_seed2.npz")["frames"]
    nf = len(Ah)

    uvjA, _ = cam.project(Pj[:nf, 0]); uvjB, _ = cam.project(Pj[:nf, 1])
    uvdA, _ = cam.project(Pd[:nf, 0]); uvdB, _ = cam.project(Pd[:nf, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    ax = axes[0]
    ax.imshow(frames[0]); ax.axis("off")
    mA = ~np.isnan(Ah[:, 0]); mB = ~np.isnan(Bh[:, 0])
    ax.plot(Ah[mA, 0], Ah[mA, 1], "o", ms=3.5, color="#ff3b30", label="observed (held-out video)")
    ax.plot(Bh[mB, 0], Bh[mB, 1], "o", ms=3.5, color="#4d86ff")
    ax.plot(uvjA[:, 0], uvjA[:, 1], "-", lw=2.6, color="#7fd18b", label=f"predicted — recovered material ({r_j:.0f} px)")
    ax.plot(uvjB[:, 0], uvjB[:, 1], "-", lw=2.6, color="#7fd18b")
    ax.plot(uvdA[:, 0], uvdA[:, 1], "--", lw=2.2, color="#f2a53d", label=f"predicted — default material ({r_d:.0f} px)")
    ax.plot(uvdB[:, 0], uvdB[:, 1], "--", lw=2.2, color="#f2a53d")
    ax.legend(fontsize=8.5, facecolor="#1d1a15", edgecolor="#443c30", labelcolor="#f2ece2", loc="upper left")
    ax.set_title("Predicting a generated video we never fitted", color="#f2ece2", fontsize=13, weight="bold")

    ax2 = axes[1]; ax2.set_facecolor(BG)
    bars = ["recovered\nmaterial", "default\nmaterial"]
    vals = [r_j, r_d]
    ax2.bar(bars, vals, color=["#7fd18b", "#f2a53d"], width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v + max(vals) * 0.03, f"{v:.1f} px", ha="center", color="#f2ece2", fontsize=13, weight="bold")
    ax2.set_ylabel("prediction error on the held-out video (px)", color="#c9bfae")
    for s in ax2.spines.values():
        s.set_color("#443c30")
    ax2.tick_params(colors="#a3988a")
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_title(f"jointly recovered:  cd {th['cd']:.1f} · μ {th['mu']:.2f} · mass ratio {th['ratio']:.2f}",
                  color="#f2ece2", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_generated.png", dpi=125, facecolor=BG)
    print(f"wrote {OUT}/heldout_generated.png   recovered {r_j:.1f}px vs default {r_d:.1f}px")


if __name__ == "__main__":
    raise SystemExit(main())
