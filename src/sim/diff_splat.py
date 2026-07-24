"""Differentiable 'squishy splat': a free cloth sheet dropped onto a ground plane.

Uses Newton's SemiImplicit cloth solver (which IS differentiable — unlike VBD/XPBD)
plus a custom differentiable ground-penalty force (same pattern as the wind force),
so the whole rollout is differentiable. The cloth's STRETCH STIFFNESS is its
'squishiness': a floppy sheet (low ke) spreads and drapes on impact, a stiff sheet
stays flat and springs. We recover log(tri_ke) from the splat by gradient descent.
"""
import numpy as np
import warp as wp

import newton


@wp.kernel
def fill_stiffness(log_ke: wp.array(dtype=float), mats: wp.array2d(dtype=float)):
    t = wp.tid()
    ke = wp.exp(log_ke[0])
    mats[t, 0] = ke        # tri_ke (stretch)
    mats[t, 1] = ke        # tri_ka (area) tied to ke


@wp.kernel
def ground_penalty(particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3),
                   ground_z: float, k: float, cd: float, mu: float,
                   particle_f: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    pen = ground_z - particle_q[i][2]
    if pen > 0.0:
        v = particle_qd[i]
        fn = wp.max(k * pen - cd * v[2], 0.0)          # upward normal penalty
        vt = wp.vec3(v[0], v[1], 0.0)
        vt_mag = wp.length(vt)
        ft = -mu * fn * vt / (vt_mag + 1.0e-3)          # regularized Coulomb friction
        wp.atomic_add(particle_f, i, wp.vec3(ft[0], ft[1], fn))


@wp.kernel
def accum_sq(pred: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3),
             scale: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    d = pred[i] - target[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class DiffSplat:
    def __init__(self, dim=16, cell=0.035, mass=0.015, height=1.05, ground_z=0.706,
                 tri_ke=800.0, tri_kd=6.0, edge_ke=2.0, k=4000.0, cd=6.0, mu=0.4,
                 tilt_deg=55.0, fps=60, num_frames=54, substeps=32, requires_grad=True):
        self.fps, self.num_frames, self.substeps = fps, num_frames, substeps
        self.sim_dt = 1.0 / (fps * substeps)
        self.ground_z, self.k, self.cd, self.mu = float(ground_z), float(k), float(cd), float(mu)
        rg = requires_grad

        b = newton.ModelBuilder()
        b.default_particle_radius = 0.005
        th = np.radians(tilt_deg)
        tilt = wp.quat(float(np.sin(th / 2)), 0.0, 0.0, float(np.cos(th / 2)))  # tilt about x
        b.add_cloth_grid(pos=wp.vec3(-dim * cell / 2, -dim * cell / 3, height),
                         rot=tilt, vel=wp.vec3(0.0, 0.0, 0.0),  # tilted -> folds/collapses on impact
                         dim_x=dim, dim_y=dim, cell_x=cell, cell_y=cell, mass=mass,
                         tri_ke=tri_ke, tri_ka=tri_ke, tri_kd=tri_kd, edge_ke=edge_ke, edge_kd=0.1)
        self.model = b.finalize(requires_grad=rg)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.solver.enable_tri_contact = False
        self.n = self.model.particle_count

        self.log_ke = wp.array([np.log(tri_ke)], dtype=float, requires_grad=rg)
        self.states = [self.model.state() for _ in range(num_frames * substeps + 1)]
        self.control = self.model.control()
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_log_ke(self, v):
        self.log_ke.assign(np.array([float(v)], np.float32))

    def rollout(self):
        wp.launch(fill_stiffness, self.model.tri_count, inputs=[self.log_ke],
                  outputs=[self.model.tri_materials])
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
