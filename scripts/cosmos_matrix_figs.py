"""Figures for the probe->parameter matrix run on Cosmos3-GENERATED video.

Cell evidence is the EFFECT SIZE: how far the best achievable fit moves (in px) when
the parameter is swept across its whole prior, with the unknown launch velocity
re-fitted at every step. Large effect => the video constrains the parameter; an effect
below the tracking noise => the video says nothing about it. This is the same
quantity that carried the synthetic matrix, and it does not depend on knowing truth.

Reads outputs/probes_i2v/cosmos_matrix.npz (from cosmos_matrix.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "probes_i2v"
BG = "#141210"
NOISE = 2.5
PROBES = ["drop", "push", "collide"]
NICE = {"drop": "Drop\nball falls on the table",
        "push": "Push\nball rolls and slows",
        "collide": "Collide\nball knocks the other"}
PLAB = [("cd", "Restitution"), ("mu", "Friction"), ("ratio", "Mass ratio")]
# baseline fit residual per probe, from the logged cosmos_matrix.py run
BASE_RMS = {"drop": 9.9, "push": 34.3, "collide": 27.7}
# the push video's path bends 57 px (17% of its length) off a straight line: a ball
# rolling on a flat table has no lateral force, so that motion is not physical
REJECTED = {"push": "motion not physical — path curves 17% off a straight line"}


def load():
    d = np.load(OUT / "cosmos_matrix.npz", allow_pickle=True)
    return {k: json.loads(v) for k, v in zip(d["keys"], d["vals"])}


def verdict(r, probe):
    """A railed optimum (sitting on the edge of the prior) is only a one-sided BOUND —
    the real value may lie outside what we scanned, so it is never 'recovered'."""
    if probe in REJECTED:
        return "rejected"
    if r["spread"] < NOISE:
        return "invisible"
    if r.get("railed"):
        return "bounded"
    if r["identified"] or (r["spread"] > 8 * NOISE and r["frac"] < 0.3):
        return "recovered"
    return "weak"


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = load()
    COL = {"recovered": "#7fd18b", "weak": "#c8b56b", "bounded": "#c8b56b",
           "invisible": "#3a332b", "rejected": "#4a2b2b"}
    FG = {"recovered": "#10331a", "weak": "#2e2712", "bounded": "#2e2712",
          "invisible": "#f2ece2", "rejected": "#f0cfcf"}

    fig, ax = plt.subplots(figsize=(10.4, 5.6), facecolor=BG); ax.set_facecolor(BG)
    ax.set_xlim(-0.5, len(PLAB) - 0.5); ax.set_ylim(len(PROBES) - 0.5, -0.5)
    for i, p in enumerate(PROBES):
        for j, (k, lab) in enumerate(PLAB):
            r = R[f"{p}|{k}"]; v = verdict(r, p)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=COL[v], edgecolor=BG, lw=2))
            head = {"recovered": "✓ recovered", "weak": "~ weakly seen", "bounded": "~ bounded only",
                    "invisible": "✗ invisible", "rejected": "— probe rejected"}[v]
            ax.text(j, i - 0.20, head, ha="center", va="center", color=FG[v], fontsize=13, weight="bold")
            if v == "recovered":
                sub = f"mass ratio ≈ {r['best']:.2f}" if k == "ratio" else f"≈ {r['best']:.2f}"
            elif v in ("weak", "bounded"):
                at_low = abs(r["best"] - min(r["scan"])) < 1e-9
                sub = (f"only: ≤ {r['hi']:.2f}" if at_low else
                       (f"only: ≥ {r['lo']:.2f}" if abs(r["best"] - max(r["scan"])) < 1e-9
                        else f"only: {r['lo']:.2f} – {r['hi']:.2f}"))
            elif v == "invisible":
                sub = "any value fits equally"
            else:
                sub = "nothing recoverable"
            ax.text(j, i + 0.05, sub, ha="center", va="center", color=FG[v], fontsize=10.5)
            ax.text(j, i + 0.27, f"changes the fit by {r['spread']:.1f} px", ha="center", va="center",
                    color=FG[v], fontsize=9, alpha=0.75)
    ax.set_xticks(range(len(PLAB))); ax.set_xticklabels([l for _, l in PLAB], color="#f2ece2", fontsize=13, weight="bold")
    ax.set_yticks(range(len(PROBES)))
    ax.set_yticklabels([NICE[p] for p in PROBES], color="#f2ece2", fontsize=11, weight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, p in enumerate(PROBES):
        ax.text(len(PLAB) - 0.42, i, f"fit\n{BASE_RMS[p]:.0f} px", ha="left", va="center",
                color=("#ef6a5a" if BASE_RMS[p] > 20 else "#7fd18b"), fontsize=9, weight="bold")
    fig.text(0.5, 0.955, "Physics from AI-generated video — what each experiment reveals",
             ha="center", color="#f2ece2", fontsize=15.5, weight="bold")
    fig.text(0.5, 0.90, "three Cosmos3 videos of the same real scene · no ground truth, so each cell reports how much that parameter changes the fit",
             ha="center", color="#a3988a", fontsize=9.5)
    fig.text(0.5, 0.022, "mass is invisible until the balls collide (0.0 px → 72.8 px) — the same structure measured in simulation",
             ha="center", color="#c9bfae", fontsize=10.5, weight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.875])
    fig.savefig(OUT / "cosmos_matrix.png", dpi=125, facecolor=BG)

    # profile curves
    fig2, axes = plt.subplots(len(PROBES), len(PLAB), figsize=(11, 7), facecolor=BG)
    for i, p in enumerate(PROBES):
        for j, (k, lab) in enumerate(PLAB):
            a = axes[i, j]; r = R[f"{p}|{k}"]; a.set_facecolor(BG)
            sc = np.array(r["scan"]); rm = np.array(r["rms"])
            col = COL[verdict(r, p)] if verdict(r, p) != "invisible" else "#8a8072"
            a.plot(sc, rm, "-o", ms=3.5, color=col, lw=2)
            if k != "mu":
                a.set_xscale("log")
            for s in a.spines.values():
                s.set_color("#443c30")
            a.tick_params(colors="#a3988a", labelsize=8)
            if i == 0:
                a.set_title(lab, color="#f2ece2", fontsize=12, weight="bold")
            if j == 0:
                a.set_ylabel(p, color="#c9bfae", fontsize=11, weight="bold")
    fig2.suptitle("Why: a dip means the video pins that parameter down; flat means it cannot",
                  color="#f2ece2", fontsize=13, weight="bold")
    fig2.text(0.5, 0.015, "fit error (px) as each parameter is swept, launch velocity re-fitted at every point",
              ha="center", color="#a3988a", fontsize=9.5)
    fig2.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig2.savefig(OUT / "cosmos_profiles.png", dpi=120, facecolor=BG)
    print(f"wrote {OUT}/cosmos_matrix.png + cosmos_profiles.png")
    for p in PROBES:
        for k, lab in PLAB:
            r = R[f"{p}|{k}"]
            print(f"  {p:8s} {lab:11s} {verdict(r,p):10s} spread {r['spread']:5.1f}px best {r['best']:6.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
