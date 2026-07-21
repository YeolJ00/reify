"""M3: full-theta differentiable flag sim.

theta is a single wp.array of 8 floats (one grad array), layout:
    0..2  wind Fourier coeffs a0, a1, a2 (m/s):  s(t) = a0 + a1 sin(wt) + a2 cos(wt)
    3     gravity_z (m/s^2)
    4     log(tri_ke)   (tri_ka tied to tri_ke)
    5     log(tri_kd)
    6     log(edge_ke)  (edge_kd stays at its config value)
    7     log(per-particle mass)

Positive parameters live in log space so optimization is unconstrained.
Model arrays (tri_materials, edge_bending_properties, particle_mass/inv_mass,
gravity) are (re)filled from theta by warp kernels inside the tape, so adjoints
flow from the solver kernels back to the theta vector. Verified against
newton 1.4.0: solvers read gravity/materials from these arrays at every launch.
"""

import numpy as np
import warp as wp

from .rollout import FlagSim

THETA_NAMES = ["wind_a0", "wind_a1", "wind_a2", "gravity_z", "log_tri_ke", "log_tri_kd", "log_edge_ke", "log_mass"]
THETA_DIM = 8


def theta_vec_from_cfg(theta_cfg: dict) -> np.ndarray:
    """Build the 8-vector from a config dict like cfg['theta']['true']."""
    # float() casts guard against pyyaml parsing "5.0e3" (no sign) as a string
    return np.array(
        [
            float(theta_cfg["wind"][0]),
            float(theta_cfg["wind"][1]),
            float(theta_cfg["wind"][2]),
            float(theta_cfg["gravity_z"]),
            np.log(float(theta_cfg["tri_ke"])),
            np.log(float(theta_cfg["tri_kd"])),
            np.log(float(theta_cfg["edge_ke"])),
            np.log(float(theta_cfg["mass"])),
        ],
        dtype=np.float64,
    )


def theta_natural(vec: np.ndarray) -> dict:
    """Human-readable natural units."""
    return {
        "wind_a0": vec[0],
        "wind_a1": vec[1],
        "wind_a2": vec[2],
        "gravity_z": vec[3],
        "tri_ke": float(np.exp(vec[4])),
        "tri_kd": float(np.exp(vec[5])),
        "edge_ke": float(np.exp(vec[6])),
        "mass": float(np.exp(vec[7])),
    }


@wp.kernel
def fill_tri_materials_kernel(theta: wp.array(dtype=float), mats: wp.array2d(dtype=float)):
    tid = wp.tid()
    ke = wp.exp(theta[4])
    mats[tid, 0] = ke
    mats[tid, 1] = ke
    mats[tid, 2] = wp.exp(theta[5])


@wp.kernel
def fill_edge_props_kernel(theta: wp.array(dtype=float), props: wp.array2d(dtype=float)):
    tid = wp.tid()
    props[tid, 0] = wp.exp(theta[6])


@wp.kernel
def fill_mass_kernel(
    theta: wp.array(dtype=float),
    free_mask: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
):
    tid = wp.tid()
    m = wp.exp(theta[7])
    mass[tid] = m * free_mask[tid]
    inv_mass[tid] = free_mask[tid] / m


@wp.kernel
def fill_gravity_kernel(theta: wp.array(dtype=float), gravity: wp.array(dtype=wp.vec3)):
    gravity[0] = wp.vec3(0.0, 0.0, theta[3])


@wp.kernel
def wind_force_theta_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    theta: wp.array(dtype=float),
    wind_dir: wp.vec3,
    drag_coeff: float,
    omega_t: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    strength = theta[0] + theta[1] * wp.sin(omega_t) + theta[2] * wp.cos(omega_t)

    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]

    p0 = particle_q[i]
    p1 = particle_q[j]
    p2 = particle_q[k]

    n = wp.cross(p1 - p0, p2 - p0)
    area = 0.5 * wp.length(n) + 1.0e-8
    n_hat = n / (2.0 * area)

    v_tri = (particle_qd[i] + particle_qd[j] + particle_qd[k]) / 3.0
    v_rel = strength * wind_dir - v_tri

    f = drag_coeff * area * wp.dot(n_hat, v_rel) * n_hat / 3.0

    wp.atomic_add(particle_f, i, f)
    wp.atomic_add(particle_f, j, f)
    wp.atomic_add(particle_f, k, f)


class ThetaFlagSim(FlagSim):
    """FlagSim whose full physics vector theta is a differentiable input."""

    def __init__(self, cfg: dict, requires_grad: bool = False):
        super().__init__(cfg, requires_grad=requires_grad)

        self.wind_freq_hz = float(cfg["theta"]["wind_freq_hz"])
        theta0 = theta_vec_from_cfg(cfg["theta"]["init_values"])
        self.theta = wp.array(theta0.astype(np.float32), dtype=float, requires_grad=requires_grad)

        # 1.0 for dynamic particles, 0.0 for pinned — fixed, from the builder's masses
        self.free_mask = wp.array(
            (self.model.particle_inv_mass.numpy() > 0.0).astype(np.float32),
            dtype=float,
            requires_grad=False,
        )

    def set_theta(self, vec: np.ndarray):
        assert vec.shape == (THETA_DIM,)
        self.theta.assign(vec.astype(np.float32))

    def theta_np(self) -> np.ndarray:
        return self.theta.numpy().astype(np.float64)

    def apply_theta(self):
        """Write theta into the model arrays. Call inside the tape."""
        m = self.model
        wp.launch(fill_tri_materials_kernel, dim=m.tri_count, inputs=[self.theta], outputs=[m.tri_materials])
        wp.launch(fill_edge_props_kernel, dim=m.edge_count, inputs=[self.theta], outputs=[m.edge_bending_properties])
        wp.launch(
            fill_mass_kernel,
            dim=m.particle_count,
            inputs=[self.theta, self.free_mask],
            outputs=[m.particle_mass, m.particle_inv_mass],
        )
        wp.launch(fill_gravity_kernel, dim=1, inputs=[self.theta], outputs=[m.gravity])

    def rollout(self):
        self.apply_theta()
        two_pi_f = 2.0 * np.pi * self.wind_freq_hz
        for t in range(self.num_frames * self.substeps):
            s_in, s_out = self.states[t], self.states[t + 1]
            s_in.clear_forces()
            wp.launch(
                wind_force_theta_kernel,
                dim=self.model.tri_count,
                inputs=[
                    s_in.particle_q,
                    s_in.particle_qd,
                    self.model.tri_indices,
                    self.theta,
                    self.wind_dir,
                    self.drag_coeff,
                    float(two_pi_f * t * self.sim_dt),
                ],
                outputs=[s_in.particle_f],
            )
            self.solver.step(s_in, s_out, self.control, self.contacts, self.sim_dt)
