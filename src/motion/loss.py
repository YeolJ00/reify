"""Trajectory loss: mean squared vertex distance to a synthetic target rollout.

Accumulated frame-by-frame into a single wp scalar so the whole thing sits on
one wp.Tape with the rollout.
"""

import numpy as np
import warp as wp


@wp.kernel
def accum_sq_dist_kernel(
    pred: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    scale: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    d = pred[tid] - target[tid]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class TrajectoryLoss:
    """MSE over (frames x particles x 3) between rollout and a fixed target."""

    def __init__(self, target_traj: np.ndarray, requires_grad: bool = True):
        # target_traj: (num_frames+1, n, 3); frame 0 is identical by construction
        self.targets = [
            wp.array(f, dtype=wp.vec3, requires_grad=False) for f in target_traj
        ]
        self.n = target_traj.shape[1]
        self.num_frames = target_traj.shape[0] - 1
        self.scale = 1.0 / (self.num_frames * self.n)
        self.loss = wp.zeros(1, dtype=float, requires_grad=requires_grad)

    def compute(self, frame_states):
        """Accumulate loss over frames 1..T. Call inside the tape, after rollout."""
        self.loss.zero_()
        for f in range(1, self.num_frames + 1):
            wp.launch(
                accum_sq_dist_kernel,
                dim=self.n,
                inputs=[frame_states[f].particle_q, self.targets[f], self.scale],
                outputs=[self.loss],
            )
        return self.loss

    def value(self) -> float:
        return float(self.loss.numpy()[0])
