"""Differentiable rebuild of the judge's video preprocessing.

G1a gave ds/d(pixel_values_videos) -- the gradient with respect to the PATCH tensor the
model consumes, not with respect to image pixels. To attribute that back to the simulated
object we need the gradient in image space, so the preprocessing between RGB frames and
patches has to be differentiable rather than a numpy black box inside the processor.

This is not a differentiable RENDERER. It is the bridge that lets one exist cheaply: with
it, the leaf of the autograd graph is a (T, H, W, 3) frame tensor, so ds/d(frame pixels)
falls out of the same backward pass that already works.

Replicates Qwen3VLVideoProcessor._preprocess exactly: rescale by 1/255, normalise with
mean/std 0.5 (so the range is [-1, 1]), pad the time axis to a multiple of the temporal
patch size by repeating the last frame, then the view/permute/reshape that builds patches.
verify() checks the result against the real processor rather than trusting the reading.
"""
import torch

PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
MEAN = 0.5
STD = 0.5


def frames_to_patches(frames, patch=PATCH, temporal_patch=TEMPORAL_PATCH, merge=MERGE):
    """(T, H, W, 3) float in [0,255] -> (n_patches, 3*temporal*patch*patch), differentiable.

    H and W must already be multiples of patch*merge; the render is authored that way so
    no resize is needed and none is applied. A resize here would add an interpolation the
    processor does not perform, which would silently shift every gradient.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3), got {tuple(frames.shape)}")
    T, H, W, _ = frames.shape
    step = patch * merge
    if H % step or W % step:
        raise ValueError(f"H,W must be multiples of {step}; got {H}x{W}")

    x = frames.permute(0, 3, 1, 2)                       # T,C,H,W
    x = (x / 255.0 - MEAN) / STD

    pad = (-T) % temporal_patch
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)], dim=0)
    grid_t = x.shape[0] // temporal_patch
    grid_h, grid_w = H // patch, W // patch
    C = 3

    x = x.reshape(grid_t, temporal_patch, C,
                  grid_h // merge, merge, patch,
                  grid_w // merge, merge, patch)
    # same permutation as Qwen3VLVideoProcessor, minus the batch axis
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(grid_t * grid_h * grid_w,
                     C * temporal_patch * patch * patch)


def verify(processor, frames_uint8, atol=1e-4):
    """Compare against the real processor. Returns (ok, max_abs_diff, shapes)."""
    import numpy as np
    ref = processor(text=["x"], images=None, videos=[frames_uint8],
                    return_tensors="pt")["pixel_values_videos"].float()
    mine = frames_to_patches(torch.as_tensor(np.asarray(frames_uint8), dtype=torch.float32))
    if ref.shape != mine.shape:
        return False, float("inf"), (tuple(ref.shape), tuple(mine.shape))
    d = (ref - mine).abs().max().item()
    return d <= atol, d, (tuple(ref.shape), tuple(mine.shape))
