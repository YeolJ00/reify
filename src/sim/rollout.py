"""Differentiable cloth-flag rollout with a constant-wind force on faces.

Wind model (aerodynamic normal pressure, standard cloth wind):
    per triangle:  f = drag * area * dot(n_hat, v_wind - v_tri) * n_hat
distributed equally to the triangle's three vertices via atomic adds into
state.particle_f before each solver substep. v_wind = strength * dir, and
`strength` is a wp.array of length 1 so it can carry gradients (theta).

All states for the full rollout are pre-allocated so a single wp.Tape can
record the whole trajectory (pattern taken from newton's diffsim examples).
"""

import numpy as np
import warp as wp

import newton

from .scene import build_flag_model


@wp.kernel
def wind_force_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    wind_strength: wp.array(dtype=float),
    wind_dir: wp.vec3,
    drag_coeff: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]

    p0 = particle_q[i]
    p1 = particle_q[j]
    p2 = particle_q[k]

    n = wp.cross(p1 - p0, p2 - p0)  # area-weighted normal (|n| = 2*area)
    area = 0.5 * wp.length(n) + 1.0e-8
    n_hat = n / (2.0 * area)

    v_tri = (particle_qd[i] + particle_qd[j] + particle_qd[k]) / 3.0
    v_rel = wind_strength[0] * wind_dir - v_tri

    f = drag_coeff * area * wp.dot(n_hat, v_rel) * n_hat / 3.0

    wp.atomic_add(particle_f, i, f)
    wp.atomic_add(particle_f, j, f)
    wp.atomic_add(particle_f, k, f)


class FlagSim:
    """Owns the model, solver, pre-allocated states, and the wind parameter."""

    def __init__(self, cfg: dict, requires_grad: bool = False):
        self.cfg = cfg
        self.requires_grad = requires_grad

        sim = cfg["sim"]
        self.fps = int(sim["fps"])
        self.num_frames = int(sim["num_frames"])
        self.substeps = int(sim["substeps"])
        self.sim_dt = 1.0 / (self.fps * self.substeps)
        self.solver_name = str(sim["solver"])

        self.model = build_flag_model(cfg, requires_grad=requires_grad)

        if self.solver_name == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=int(sim["vbd_iterations"]))
        elif self.solver_name == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.solver.enable_tri_contact = False
        else:
            raise ValueError(f"unknown solver '{self.solver_name}'")

        w = cfg["wind"]
        d = np.asarray(w["direction"], dtype=np.float64)
        d = d / np.linalg.norm(d)
        self.wind_dir = wp.vec3(*d.tolist())
        self.drag_coeff = float(w["drag_coeff"])
        self.wind_strength = wp.array(
            [float(w["strength"])], dtype=float, requires_grad=requires_grad
        )

        total_substeps = self.num_frames * self.substeps
        self.states = [self.model.state() for _ in range(total_substeps + 1)]
        self.control = self.model.control()
        self.contacts = None

    def set_wind_strength(self, value: float):
        self.wind_strength.assign(np.array([value], dtype=np.float32))

    def rollout(self):
        """Run the full rollout. Call inside a wp.Tape() to record gradients."""
        for t in range(self.num_frames * self.substeps):
            s_in, s_out = self.states[t], self.states[t + 1]
            s_in.clear_forces()
            wp.launch(
                wind_force_kernel,
                dim=self.model.tri_count,
                inputs=[
                    s_in.particle_q,
                    s_in.particle_qd,
                    self.model.tri_indices,
                    self.wind_strength,
                    self.wind_dir,
                    self.drag_coeff,
                ],
                outputs=[s_in.particle_f],
            )
            self.solver.step(s_in, s_out, self.control, self.contacts, self.sim_dt)

    def frame_states(self):
        """States at frame boundaries (num_frames + 1 entries, incl. initial)."""
        return [self.states[t * self.substeps] for t in range(self.num_frames + 1)]

    def trajectory(self) -> np.ndarray:
        """Vertex positions at frame boundaries: (num_frames+1, n_particles, 3)."""
        return np.stack([s.particle_q.numpy() for s in self.frame_states()])
