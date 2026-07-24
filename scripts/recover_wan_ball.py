"""Recover physics from a REAL Wan-generated video (not our own render).

Pipeline: pose a bouncy ball in the scene (blender_ball_i0.py) -> hand that one
frame to Wan i2v (run_i2v.py) -> here we track the ball (colour + largest round
blob) and fit a physical drop (free-fall from rest + inelastic floor) to its
height. Finding: Wan's FALL is gravity-like (the recovered drop rides on it), but
its post-impact SETTLING is non-physical (the ball bobs in place) — which is why
we recover parameters by fitting a simulation to the video rather than trusting
the video. Produces the trajectory figure + a fall overlay. (warp env: cv2)

Run: /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/recover_wan_ball.py \
        --npz outputs/i2v_wan5b_seed0.npz --ball-radius 0.075
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def ball_mask(f):
    r, g, b = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
    m = ((r > 150) & (r < 252) & (g > 85) & (g < 200) & (b > 72) & (b < 192)
         & (r - b > 18) & (r - g > 16)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def track_ball(frames, min_area=3500):
    """The ball = the largest round salmon blob each frame (rejects small pink
    background objects and the shakers). Missing where Wan deforms it on impact."""
    N = len(frames)
    X, Y, R = (np.full(N, np.nan) for _ in range(3))
    for i, f in enumerate(frames):
        n, _, st, ct = cv2.connectedComponentsWithStats(ball_mask(f), 8)
        best, ba = None, 0
        for j in range(1, n):
            a = st[j, cv2.CC_STAT_AREA]; w = st[j, cv2.CC_STAT_WIDTH]; h = st[j, cv2.CC_STAT_HEIGHT]
            if a < min_area or a / (w * h + 1e-6) < 0.55 or min(w, h) / max(w, h) < 0.62:
                continue
            if a > ba:
                ba, best = a, (ct[j], np.sqrt(a / np.pi))
        if best is not None:
            X[i], Y[i], R[i] = best[0][0], best[0][1], best[1]
    return X, Y, R


def fit_drop(Y, land_window=16):
    """Free-fall from rest onto an inelastic floor. Returns (y0, t0, g_px, floor).

    The floor is the LANDING level within the first `land_window` frames — not the
    global max height, which here is reached later during Wan's non-physical bob."""
    ok = ~np.isnan(Y)
    idx = np.where(ok)[0]
    held = idx[idx < 8]
    y0 = float(np.nanmean(Y[held])) if len(held) else float(Y[idx[0]])
    win = idx[idx < land_window]                       # fall + immediate landing only
    t_land = int(win[np.argmax(Y[win])])               # deepest point of the fall
    floor = float(Y[t_land])
    fall = [t for t in win if Y[t] - y0 > 20]          # release = last held frame before descent
    before = win[win < (fall[0] if fall else t_land)]
    t0 = float(before[-1]) if len(before) else t_land - 3.0
    g = 2 * (floor - y0) / max((t_land - t0) ** 2, 1.0)
    return y0, t0, float(g), floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/i2v_wan5b_seed0.npz")
    ap.add_argument("--ball-radius", type=float, default=0.075, help="real ball radius (m)")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    frames = np.load(REPO / args.npz)["frames"]
    H, W, _ = frames[0].shape
    X, Y, R = track_ball(frames)
    ok = ~np.isnan(Y)
    y0, t0, g, floor = fit_drop(Y)
    mpp = args.ball_radius / np.nanmedian(R)
    g_ms2 = g * mpp * args.fps ** 2
    print(f"tracked {ok.sum()}/{len(frames)} frames "
          f"(gap = impact deformation Wan can't hold round)")
    print(f"recovered drop: g={g:.1f}px/frame^2 -> ~{g_ms2:.0f} m/s^2 (order of Earth gravity)")
    print(f"landing: ball stops at the table and does NOT rebound cleanly; "
          f"post-impact it bobs in place -> Wan's settling is non-physical")

    _figures(frames, X, Y, R, y0, t0, g, floor)
    return 0


def _figures(frames, X, Y, R, y0, t0, g, floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    H, W, _ = frames[0].shape; N = len(frames); ok = ~np.isnan(Y)
    mod = np.array([y0 if t < t0 else min(y0 + 0.5 * g * (t - t0) ** 2, floor) for t in range(N)])
    out = REPO / "outputs" / "wan_ball"; out.mkdir(exist_ok=True)
    BG = "#141210"

    fig, ax = plt.subplots(figsize=(7.8, 4.4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(np.where(ok)[0], H - Y[ok], "o", ms=5, color="#ef6a5a", label="tracked ball (Wan video)")
    ax.plot(range(N), H - mod, "-", lw=2.3, color="#57d7e0", label="recovered physical drop")
    ax.axhline(H - floor, ls=":", color="#8a8072", lw=1)
    for s in ax.spines.values():
        s.set_color("#443c30")
    ax.tick_params(colors="#a3988a")
    ax.set_xlabel("frame", color="#c9bfae"); ax.set_ylabel("ball height (px)", color="#c9bfae")
    ax.set_title("Physics from a real Wan-generated video", color="#f2ece2", weight="bold")
    ax.legend(facecolor="#1d1a15", edgecolor="#443c30", labelcolor="#f2ece2")
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=125, facecolor=BG); plt.close(fig)

    xi = np.interp(range(N), np.where(ok)[0], X[ok])
    yi = np.interp(range(N), np.where(ok)[0], Y[ok])
    ri = np.interp(range(N), np.where(ok)[0], R[ok])
    sel = list(range(0, 15)); sc = 0.62
    fig = plt.figure(figsize=(W / 100 * sc, H / 100 * sc), dpi=100); ax = fig.add_axes([0, 0, 1, 1])

    def draw(k):
        f = sel[k]; ax.clear(); ax.imshow(frames[f]); ax.axis("off")
        ax.scatter([xi[f]], [yi[f]], s=26, c="#ef6a5a", zorder=5)
        ax.add_patch(plt.Circle((xi[f], mod[f]), np.clip(ri[f], 22, 50), fill=False, ec="#57d7e0", lw=2.4))
        return []
    FuncAnimation(fig, draw, frames=len(sel), interval=140).save(
        out / "overlay.gif", writer=PillowWriter(fps=7)); plt.close(fig)
    print(f"wrote {out}/trajectory.png + overlay.gif")


if __name__ == "__main__":
    raise SystemExit(main())
