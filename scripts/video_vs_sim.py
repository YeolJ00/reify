"""Side-by-side: the video Cosmos generated vs the simulation we recovered from it.

Left  — the generated clip (what the world model imagined).
Right — our simulator, running the material that was fitted from those clips, drawn on
        the same static backdrop through the same camera.

This is the honest visual for "the video model supplies the what, the simulator supplies
the guarantee": the right-hand side is executable physics, and you can see where it
agrees with the imagination and where the imagination drifts off physics.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/video_vs_sim.py     (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from src.track.balls import track_balls  # noqa: E402
from scripts.cosmos_joint_heldout import OUT, simulate  # noqa: E402

CLIPS = [("drop", 0, "Drop"), ("collide", 2, "Collide")]
COL = {"A": np.array([0.86, 0.20, 0.16]), "B": np.array([0.24, 0.42, 0.86])}


def ball_patch(ax, uv, depth, R, fx, color, light_from_left=True):
    """Draw a shaded disc for a ball at projected uv with the right apparent size."""
    import matplotlib.pyplot as plt
    r = fx * R / max(depth, 1e-6)
    ax.add_patch(plt.Circle(uv, r, color=color * 0.55, zorder=4))
    ax.add_patch(plt.Circle(uv + np.array([-r * 0.22, -r * 0.22]), r * 0.72,
                            color=np.clip(color * 1.25, 0, 1), zorder=5))


def main():
    cfg = json.loads((OUT / "scene.json").read_text())
    cam = Camera(cfg["camera"]); R = cfg["ball_radius"]
    jd = np.load(OUT / "joint_heldout.npz", allow_pickle=True)
    th = json.loads(str(jd["theta"])); v0s = jd["v0s"]
    v0_of = {"drop": v0s[0], "collide": v0s[1]}
    print(f"material: cd={th['cd']:.2f} mu={th['mu']:.3f} ratio={th['ratio']:.3f}")

    wp.init()
    data = []
    with wp.ScopedDevice("cuda:0"):
        for probe, seed, label in CLIPS:
            fr = np.load(OUT / f"vid_{probe}_seed{seed}.npz")["frames"]
            A, RA, B, RB = track_balls(fr)
            P = simulate(cfg, probe, v0_of[probe], th, len(fr))
            data.append((label, fr, A, B, P))
            print(f"  {label}: {len(fr)} frames simulated with the recovered material")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageSequence

    n = len(data)
    fig, axes = plt.subplots(n, 2, figsize=(7.8, 2.95 * n), facecolor="#141210")
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03, hspace=0.12)
    N = min(len(d[1]) for d in data)
    # show EVERY frame at the source rate — decimating to a third made the motion
    # unreadable. (The source itself is 24fps, so a 0.3s bounce is only ~7 frames;
    # that is a genuine limit of the generated clip, not of the playback.)
    sel = list(range(N))

    def draw(k):
        t = sel[k]
        for row, (label, fr, A, B, P) in enumerate(data):
            axL, axR = axes[row, 0], axes[row, 1]
            axL.clear(); axL.imshow(fr[t]); axL.axis("off")
            axL.set_title(f"{label} — generated video", color="#f2ece2", fontsize=11, weight="bold")
            # right: our physics on the same backdrop (frame 0 = static scene)
            axR.clear(); axR.imshow(fr[0]); axR.axis("off")
            for bi, key in enumerate(("A", "B")):
                uv, dep = cam.project(P[t:t + 1, bi])
                ball_patch(axR, uv[0], float(dep[0]), R, cam.fx, COL[key])
            axR.set_title(f"{label} — our recovered simulation", color="#7fd18b", fontsize=11, weight="bold")
        return []

    # render each frame explicitly (FuncAnimation+PillowWriter produced sheared frames
    # here — a frame-buffer stride mismatch); this guarantees identical, correct frames
    import io
    frames_png = []
    for k in range(len(sel)):
        draw(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=86, facecolor="#141210")
        buf.seek(0)
        frames_png.append(Image.open(buf).convert("RGB"))
    p = OUT / "video_vs_sim.gif"
    size = frames_png[0].size
    fs = [f.resize(size).convert("P", palette=Image.ADAPTIVE, colors=72) for f in frames_png]
    fs[0].save(p, save_all=True, append_images=fs[1:], loop=0, duration=42, optimize=True)
    print(f"wrote {p} ({p.stat().st_size // 1024} KiB, {len(fs)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
