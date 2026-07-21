"""Forward render of the cloth rollout (architecture step 2 — NOT differentiable).

Rasterizes the projected mesh with matplotlib in exact pixel coordinates,
painter-sorted by camera depth, with a static random per-triangle grayscale
texture so the Lucas-Kanade tracker has features to lock onto.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render_rollout(traj: np.ndarray, tris: np.ndarray, camera, texture_seed: int = 0) -> np.ndarray:
    """traj (F,N,3), tris (T,3) -> uint8 frames (F,H,W,3), grayscale content."""
    rng = np.random.default_rng(texture_seed)
    # mid-gray range: keeps corners detectable without saturating
    tri_shade = 0.25 + 0.65 * rng.random(len(tris))

    W, H = camera.width, camera.height
    dpi = 100
    frames = np.empty((traj.shape[0], H, W, 3), dtype=np.uint8)

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    for f in range(traj.shape[0]):
        uv, depth = camera.project(traj[f])
        tri_depth = depth[tris].mean(axis=1)
        order = np.argsort(-tri_depth)  # far first (painter's algorithm)

        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # v grows downward
        ax.set_facecolor("black")
        ax.axis("off")
        ax.tripcolor(
            uv[:, 0], uv[:, 1], tris[order],
            facecolors=tri_shade[order], cmap="gray", vmin=0.0, vmax=1.0,
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames[f] = buf
    plt.close(fig)
    return frames
