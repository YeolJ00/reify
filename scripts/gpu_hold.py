"""Hold VRAM on a GPU until released (shared-box capacity claim).

Allocates chunks up to --gib (or ~90% of free) on the given GPU and idles.
Release cleanly by creating the release file:  touch /tmp/release_gpu<N>
(N = the *physical* GPU index passed via --gpu).

Run with the video env:
  CUDA_VISIBLE_DEVICES=<N> python scripts/gpu_hold.py --gpu <N> [--gib 40]
"""

import argparse
import os
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU index (for the release file name)")
    ap.add_argument("--gib", type=float, default=None, help="GiB to hold (default ~90%% of free)")
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES maps physical -> 0
    free_b, total_b = torch.cuda.mem_get_info(dev)
    target_b = int(args.gib * (1 << 30)) if args.gib else int(free_b * 0.90)

    chunk_b = 1 << 30  # 1 GiB
    held = []
    held_b = 0
    while held_b + chunk_b <= target_b:
        try:
            held.append(torch.empty(chunk_b // 2, dtype=torch.float16, device=dev))
            held_b += chunk_b
        except torch.cuda.OutOfMemoryError:
            break
    print(f"[gpu_hold] gpu {args.gpu}: holding {held_b / (1 << 30):.1f} GiB "
          f"(pid {os.getpid()}). Release: touch /tmp/release_gpu{args.gpu}", flush=True)

    release = Path(f"/tmp/release_gpu{args.gpu}")
    while not release.exists():
        time.sleep(5)
    release.unlink(missing_ok=True)
    print(f"[gpu_hold] gpu {args.gpu}: released", flush=True)


if __name__ == "__main__":
    main()
