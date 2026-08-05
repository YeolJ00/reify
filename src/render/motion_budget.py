"""Motion budget: make sure the event is actually inside the frames the judge receives.

This exists because of a failure, not a hypothesis. Drop height was lowered 0.30 -> 0.15 m
to stop objects rolling off the table, which collapsed the whole drop-bounce-settle event
into the gap between the judge's first and second sampled frame. Measured on the clips the
CEM and G0 v2 actually scored: 1 of 9 frame deltas contained motion for the brass pot, 2 of
9 for the duck. Nine tenths of what the judge saw was a still life, and every conclusion
drawn from those clips -- including "the judge has no stable material preference" -- was
really a statement about clips with no physics in them.

Two pieces:

* event_window() picks the frame range containing the drop, the bounces and the start of
  settling, from the SIMULATED height rather than from pixels, so it is exact and costs
  nothing. Trimming to it also removes the tail where objects roll away, which is what the
  drop height was lowered to avoid -- so the trim fixes both problems at once and the drop
  can go back up.

* motion_fraction() is the guard. It measures what fraction of the frames the judge will
  receive actually differ from their predecessor, and is meant to be called on every clip
  before it is scored. A clip that fails it is not evidence about materials.

Sampling note. The judge must see a FIXED NUMBER of frames spanning the clip, not a fixed
fps. A trimmed event is ~0.5-1 s, and at the card's 4 fps that is 2-4 frames -- trading
one kind of starvation for another. Sampling N frames evenly across whatever the clip is
keeps the event fully covered and makes clips of different lengths comparable.
"""
import numpy as np

N_FRAMES = 12          # what the judge receives per clip
MOTION_EPS = 0.5       # mean |pixel delta| that counts as "something moved"
MIN_FRACTION = 0.60    # guard threshold


def event_window(z, fps=24.0, pre=2, tail=0.10, min_len=12, max_len=None):
    """Frame range [start, end) covering impact through the decay of the bounces.

    z is the simulated centroid height. Start is a couple of frames before the fastest
    downward motion; the end is where speed has decayed to `tail` of its peak and stays
    there. Working from the trajectory avoids inferring the event from the very pixels
    whose motion content we are trying to guarantee.
    """
    z = np.asarray(z, float)
    v = np.gradient(z, 1.0 / float(fps))
    i_imp = int(np.argmin(v))
    speed = np.abs(v)
    peak = float(speed[i_imp])
    if not np.isfinite(peak) or peak <= 0:
        return 0, len(z)
    quiet = speed < tail * peak
    end = len(z)
    for i in range(i_imp + 1, len(z)):
        if quiet[i:].all():
            end = i
            break
    start = max(0, i_imp - int(pre))
    if end - start < min_len:
        end = min(len(z), start + min_len)
    if max_len is not None:
        end = min(end, start + int(max_len))
    return int(start), int(end)


def sample_frames(path, n=N_FRAMES):
    """The exact frames a judge with this decoder receives: n evenly spaced over the clip."""
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(path))
    frames = [f[..., :3] for f in rd]
    rd.close()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    idx = np.unique(np.linspace(0, len(frames) - 1, int(n)).round().astype(int))
    return np.stack([frames[i] for i in idx])


def motion_fraction(path, n=N_FRAMES, eps=MOTION_EPS):
    """Fraction of consecutive sampled-frame deltas that contain motion."""
    v = sample_frames(path, n).astype(np.float32)
    d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2, 3))
    return float((d > eps).mean()), [float(x) for x in d]


def check(path, n=N_FRAMES, eps=MOTION_EPS, min_fraction=MIN_FRACTION):
    """Guard. Returns (ok, fraction, deltas). A failing clip must not be scored."""
    frac, d = motion_fraction(path, n, eps)
    return bool(frac >= min_fraction), frac, d
