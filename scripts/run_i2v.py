"""Generate motion-prior video(s) from an initial frame with a chosen i2v backend.

Run with the `video` env:
  /home/jooyeolyun/anaconda3/envs/video/bin/python scripts/run_i2v.py \
      --backend wan5b --image outputs/m4_I0.png \
      --prompt "A rectangular cloth flag pinned along its left edge to a pole, \
waving and flapping in a steady wind against a black background." \
      --seeds 0 1 2

Writes outputs/i2v_<backend>_seed<k>.mp4 + .npz (frames array for the tracker).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.video.i2v import BACKENDS, save_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="wan5b")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--size", type=int, nargs=2, default=[480, 480], metavar=("H", "W"))
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    backend = BACKENDS[args.backend]()
    out = REPO / "outputs"
    out.mkdir(exist_ok=True)

    for seed in args.seeds:
        t0 = time.time()
        frames = backend.generate(image, args.prompt, num_frames=args.num_frames,
                                  seed=seed, height=args.size[0], width=args.size[1])
        dt = time.time() - t0
        stem = out / f"i2v_{args.backend}_seed{seed}"
        np.savez_compressed(stem.with_suffix(".npz"), frames=frames, fps=backend.fps,
                            prompt=args.prompt, seed=seed)
        save_video(frames, stem.with_suffix(".mp4"), fps=backend.fps)
        print(f"seed {seed}: {frames.shape} in {dt:.0f} s -> {stem}.mp4/.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
