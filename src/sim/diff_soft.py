"""Differentiable volumetric SOFT BODY: a tetrahedral-FEM jelly cube dropped onto the
ground. Extends deformation recovery from cloth (2D surface) to a 3D solid.

Uses Newton's `add_soft_grid` (tet FEM) + the SemiImplicit solver (differentiable)
plus the same differentiable ground-penalty force as the cloth splat. The FEM shear
modulus k_mu is the 'squishiness': a soft cube (low k_mu) squashes flat and stays; a
stiff cube (high k_mu) barely deforms and springs back. We recover log(k_mu) from how
much the cube deforms on impact. (warp env.)
"""
import numpy as np
import warp as wp

import newton

from .diff_splat import accum_sq, ground_penalty  # reuse contact + trajectory-loss kernels


@wp.kernel
def fill_tet_stiffness(log_kmu: wp.array(dtype=float), lam_ratio: float, mats: wp.array2d(dtype=float)):
    t = wp.tid()
    kmu = wp.exp(log_kmu[0])
    mats[t, 0] = kmu               # k_mu (shear modulus) — the recoverable stiffness
    mats[t, 1] = kmu * lam_ratio   # k_lambda (Lame) tied to k_mu


class DiffSoft:
    def __init__(self, dim=4, cell=0.03, density=800.0, ground_z=0.706, drop_gap=0.06,
                 k_mu=6000.0, lam_ratio=2.0, k_damp=10.0, k=3000.0, cd=25.0, mu=0.5,
                 fps=60, num_frames=40, substeps=64, requires_grad=True):
        self.fps, self.num_frames, self.substeps = fps, num_frames, substeps
        self.sim_dt = 1.0 / (fps * substeps)
        self.ground_z, self.k, self.cd, self.mu = float(ground_z), float(k), float(cd), float(mu)
        self.lam_ratio = float(lam_ratio)
        rg = requires_grad

        b = newton.ModelBuilder()
        b.default_particle_radius = 0.005
        b.add_soft_grid(pos=wp.vec3(-dim * cell / 2, -dim * cell / 2, ground_z + drop_gap),
                        rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=dim, dim_y=dim, dim_z=dim, cell_x=cell, cell_y=cell, cell_z=cell,
                        density=density, k_mu=k_mu, k_lambda=k_mu * lam_ratio, k_damp=k_damp)
        self.model = b.finalize(requires_grad=rg)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.n = self.model.particle_count
        self.tris = self.model.tri_indices.numpy() if self.model.tri_count > 0 else None

        self.log_kmu = wp.array([np.log(k_mu)], dtype=float, requires_grad=rg)
        self.states = [self.model.state() for _ in range(num_frames * substeps + 1)]
        self.control = self.model.control()
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_log_kmu(self, v):
        self.log_kmu.assign(np.array([float(v)], np.float32))

    def rollout(self):
        wp.launch(fill_tet_stiffness, self.model.tet_count, inputs=[self.log_kmu, self.lam_ratio],
                  outputs=[self.model.tet_materials])
        for t in range(self.num_frames * self.substeps):
            s = self.states[t]
            s.clear_forces()
            wp.launch(ground_penalty, self.n,
                      inputs=[s.particle_q, s.particle_qd, self.ground_z, self.k, self.cd, self.mu],
                      outputs=[s.particle_f])
            self.solver.step(s, self.states[t + 1], self.control, None, self.sim_dt)

    def frame_states(self):
        return [self.states[t * self.substeps] for t in range(self.num_frames + 1)]

    def trajectory(self):
        return np.stack([s.particle_q.numpy() for s in self.frame_states()])
