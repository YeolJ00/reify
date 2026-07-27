"""The probe->parameter matrix: one scene, several experiments, and what each can identify.

Same two objects on the same table, driven three ways:
    drop     — A falls onto the table
    slide    — A is pushed across the table
    collide  — A is pushed into B

For every (probe, parameter) pair we try to recover that parameter from that probe's
observation alone — the objects' camera-projected trajectories with realistic tracking
noise — and record the error. Identifiability is ALWAYS relative to measurement
precision: with a perfect model and zero noise every parameter looks recoverable, so we
add ~2 px of tracking noise (the scale CoTracker actually gives) and average over noise
draws. The result is a SPARSE matrix: restitution needs an impact, friction needs
sliding, and the mass ratio only becomes observable when the objects collide.

Then a held-out test: combine each parameter's best probe and PREDICT a fourth, unseen
probe with no refitting — against a 'default parameters' baseline.

Contact is stiff (k=4e4) on purpose: with soft penalty contact a PARKED object's static
sink depth depends on its density and leaks a fake ~13 px mass signal into the image.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/probe_matrix.py   (warp env)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.sim.probe_scene import ProbeScene  # noqa: E402

GZ = 0.706
NAMES = ["Threshold_Porcelain_Teapot_White", "Schleich_Lion_Action_Figure"]
CAM = {"eye": [0.55, -0.75, 1.05], "target": [0.02, 0.0, 0.78], "fov_deg": 52, "width": 640, "height": 500}
TRUE = {"cd": 12.0, "mu": 0.35, "ratio": 2.0}     # restitution-damping, friction, rho_B/rho_A
DEFAULT = {"cd": 30.0, "mu": 0.6, "ratio": 1.0}   # a plausible one-size-fits-all guess
RHO_A = 600.0
STRIDE = 40
K_CONTACT, DT, NSTEPS = 40000.0, 1.0e-4, 2800
NOISE_PX = 2.0          # tracking noise (CoTracker-scale) — sets the identifiability floor
N_SEEDS = 40            # noise draws (cheap: sims are precomputed)
GRID_N = 21

PARAMS = [("cd", "Restitution", (3.0, 80.0), True),
          ("mu", "Friction", (0.05, 1.2), False),
          ("ratio", "Mass ratio", (0.3, 4.0), True)]
PROBES = ["drop", "slide", "collide"]
NICE = {"drop": "Drop\n(falls on table)", "slide": "Push / slide\n(across table)",
        "collide": "Collide\n(knocks the other)"}


def build(probe, theta, rest):
    p = dict(TRUE); p.update(theta)
    layout = {
        "drop":    ([[0.0, 0.0, GZ + 0.14], [0.40, 0.40, rest[1]]], [[0, 0, 0], [0, 0, 0]]),
        "slide":   ([[-0.12, 0.0, rest[0]], [0.40, 0.40, rest[1]]], [[0.9, 0, 0], [0, 0, 0]]),
        "collide": ([[-0.10, 0.0, rest[0]], [0.08, 0.0, rest[1]]], [[1.6, 0, 0], [0, 0, 0]]),
        "heldout": ([[-0.10, -0.05, rest[0]], [0.08, 0.0, rest[1]]], [[1.5, 0.45, 0], [0, 0, 0]]),
    }[probe]
    return ProbeScene(NAMES, layout[0], layout[1], densities=(RHO_A, RHO_A * p["ratio"]),
                      cd=p["cd"], mu=p["mu"], ground_z=GZ, k=K_CONTACT, dt=DT, n_steps=NSTEPS)


def observe(probe, theta, rest, cam):
    """Clean 2D projected trajectories of both objects — what a tracker would give."""
    sc = build(probe, theta, rest)
    sc.rollout()
    P = sc.positions(STRIDE)
    return np.stack([cam.project(P[:, b])[0] for b in range(P.shape[1])], axis=1)


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        cam = Camera(CAM)
        p0 = ProbeScene(NAMES, [[0, 0, GZ + .2]] * 2, [[0, 0, 0]] * 2)
        cl, bd, r = p0.center_local.numpy(), p0.body.numpy(), p0.radius
        rest = [float(GZ + r - cl[bd == i][:, 2].min()) for i in range(2)]

        truth = {p: observe(p, {}, rest, cam) for p in PROBES}
        print(f"noise floor {NOISE_PX} px | grid {GRID_N} | {N_SEEDS} noise draws\n")

        M = np.zeros((len(PROBES), len(PARAMS)))         # median relative error %
        EFFECT = np.zeros_like(M)                        # peak image effect of the parameter (px)
        REC = {}
        rng = np.random.default_rng(0)
        for i, pr in enumerate(PROBES):
            for j, (key, lab, (lo, hi), logsc) in enumerate(PARAMS):
                grid = np.geomspace(lo, hi, GRID_N) if logsc else np.linspace(lo, hi, GRID_N)
                preds = np.stack([observe(pr, {key: float(g)}, rest, cam) for g in grid])
                # how much does this parameter move the image at all, vs the noise floor?
                EFFECT[i, j] = float(np.abs(preds - truth[pr]).max())
                errs, recs = [], []
                for s in range(N_SEEDS):
                    tgt = truth[pr] + rng.normal(0, NOISE_PX, truth[pr].shape)
                    k = int(np.argmin(((preds - tgt) ** 2).reshape(len(grid), -1).mean(1)))
                    recs.append(grid[k]); errs.append(100.0 * abs(grid[k] - TRUE[key]) / TRUE[key])
                M[i, j] = float(np.median(errs)); REC[(pr, key)] = float(np.median(recs))
                flag = "IDENTIFIED " if M[i, j] < 15 else ("weak       " if M[i, j] < 50 else "unidentif. ")
                print(f"  {pr:8s} x {lab:11s}: {flag} median err {M[i, j]:6.1f}%  "
                      f"(recovered {REC[(pr, key)]:6.3f} vs true {TRUE[key]:.3f}) | "
                      f"param moves image {EFFECT[i, j]:6.1f} px vs {NOISE_PX} px noise")

        best_probe = {PARAMS[j][0]: PROBES[int(np.argmin(M[:, j]))] for j in range(len(PARAMS))}
        combined = {k: REC[(best_probe[k], k)] for k, *_ in PARAMS}
        print("\ncombined estimate (each parameter from the probe that identifies it):")
        for k, lab, *_ in PARAMS:
            print(f"  {lab:11s} = {combined[k]:6.3f}  (true {TRUE[k]:.3f}, from '{best_probe[k]}')")

        ho_true = observe("heldout", {}, rest, cam)
        ho_rec = observe("heldout", combined, rest, cam)
        ho_def = observe("heldout", DEFAULT, rest, cam)
        e_rec = float(np.sqrt(np.mean((ho_rec - ho_true) ** 2)))
        e_def = float(np.sqrt(np.mean((ho_def - ho_true) ** 2)))
        print(f"\nHELD-OUT probe (angled push into B — never fitted):")
        print(f"  recovered params : {e_rec:6.1f} px error")
        print(f"  default  params  : {e_def:6.1f} px error  ({e_def / max(e_rec, 1e-9):.0f}x worse)")

    np.savez(REPO / "outputs" / "matrix" / "matrix.npz", M=M, EFFECT=EFFECT,
             ho_true=ho_true, ho_rec=ho_rec, ho_def=ho_def, e_rec=e_rec, e_def=e_def)
    _figures(M, EFFECT, ho_true, ho_rec, ho_def, e_rec, e_def)
    return 0


def _figures(M, EFFECT, ho_true, ho_rec, ho_def, e_rec, e_def):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    BG = "#141210"
    out = REPO / "outputs" / "matrix"; out.mkdir(parents=True, exist_ok=True)
    labels = [p[1] for p in PARAMS]
    cmap = LinearSegmentedColormap.from_list("id", ["#7fd18b", "#e8d48b", "#3a332b"])

    fig, ax = plt.subplots(figsize=(9.4, 5.0), facecolor=BG); ax.set_facecolor(BG)
    ident = M < 15
    # colour purely by identified / not — the numbers carry the nuance
    shade = np.where(ident, 0.0, 100.0)
    ax.imshow(shade, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ok = ident[i, j]
            fg, dim = ("#10331a", "#2c4a33") if ok else ("#f2ece2", "#a89e90")
            ax.text(j, i - 0.22, ("✓ recovered" if ok else "✗ invisible"), ha="center", va="center",
                    color=fg, fontsize=13, weight="bold")
            ax.text(j, i + 0.02, f"moves image {EFFECT[i, j]:.1f} px", ha="center", va="center",
                    color=fg, fontsize=10.5)
            ax.text(j, i + 0.22, (f"→ {M[i, j]:.0f}% error" if ok else
                                  f"→ below the {NOISE_PX:.0f} px noise floor"),
                    ha="center", va="center", color=dim, fontsize=9.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, color="#f2ece2", fontsize=13, weight="bold")
    ax.set_yticks(range(len(PROBES))); ax.set_yticklabels([NICE[p] for p in PROBES], color="#f2ece2", fontsize=11.5, weight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color("#443c30")
    fig.text(0.5, 0.955, "What each experiment can reveal", ha="center",
             color="#f2ece2", fontsize=16, weight="bold")
    fig.text(0.5, 0.905, "a parameter is only recoverable if changing it moves the image more than the tracking noise",
             ha="center", color="#a3988a", fontsize=10)
    fig.text(0.5, 0.02, "the push never impacts → bounciness invisible;  mass stays invisible until the objects collide",
             ha="center", color="#c9bfae", fontsize=10.5, weight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.885]); fig.savefig(out / "probe_matrix.png", dpi=125, facecolor=BG)

    fig2, ax2 = plt.subplots(figsize=(7.8, 4.7), facecolor=BG); ax2.set_facecolor(BG)
    for arr, lab, col, ls in [(ho_true, "observed (held-out experiment)", "#ef6a5a", "-"),
                              (ho_rec, f"predicted — recovered params  ({e_rec:.1f} px off)", "#7fd18b", "-"),
                              (ho_def, f"predicted — default params  ({e_def:.0f} px off)", "#f2a53d", "--")]:
        ax2.plot(arr[:, 1, 0], arr[:, 1, 1], ls, color=col, lw=2.5, label=lab)
        ax2.scatter([arr[0, 1, 0]], [arr[0, 1, 1]], color=col, s=30, zorder=5)
    ax2.invert_yaxis()
    for s in ax2.spines.values():
        s.set_color("#443c30")
    ax2.tick_params(colors="#a3988a")
    ax2.set_xlabel("image x (px)", color="#c9bfae"); ax2.set_ylabel("image y (px)", color="#c9bfae")
    ax2.set_title("Held-out test — predicting an experiment we never fitted",
                  color="#f2ece2", fontsize=13.5, weight="bold")
    ax2.legend(facecolor="#1d1a15", edgecolor="#443c30", labelcolor="#f2ece2", fontsize=9.5)
    fig2.tight_layout(); fig2.savefig(out / "heldout_prediction.png", dpi=125, facecolor=BG)
    print(f"wrote {out}/probe_matrix.png + heldout_prediction.png")


if __name__ == "__main__":
    raise SystemExit(main())
