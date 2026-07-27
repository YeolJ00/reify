"""Differentiable point-sphere drop: a ball centre under gravity + a differentiable
ground penalty (normal restitution via damping `cd`, tangential Coulomb friction `mu`).

Used to fit a 3D physical drop to 2D CoTracker tracks of a Wan-generated ball
(architecture steps 6-7): the sim's projected centre is matched to the tracked
centroid, so physics + camera projection disentangle a real bounce/roll from the
monocular image motion. Recovers launch velocity v0, restitution-damping cd, and
friction mu. Gravity is fixed (Wan's fall is ~real gravity at 24 fps). (warp env.)
"""
import numpy as np
import warp as wp


@wp.kernel
def _set_v0(v0: wp.array(dtype=wp.vec3), vel0: wp.array(dtype=wp.vec3)):
    vel0[0] = v0[0]


@wp.kernel
def _step(p: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          contact_z: float, k: float, cd: wp.array(dtype=float), mu: wp.array(dtype=float),
          g: float, dt: float,
          p1: wp.array(dtype=wp.vec3), v1: wp.array(dtype=wp.vec3)):
    pos = p[0]
    vel = v[0]
    f = wp.vec3(0.0, 0.0, g)                       # gravity (per unit mass)
    pen = contact_z - pos[2]
    if pen > 0.0:
        fn = wp.max(k * pen - cd[0] * vel[2], 0.0)  # upward normal penalty
        vt = wp.vec3(vel[0], vel[1], 0.0)
        ft = -mu[0] * fn * vt / (wp.length(vt) + 1.0e-3)
        f = f + wp.vec3(ft[0], ft[1], fn)
    nv = vel + f * dt
    v1[0] = nv
    p1[0] = pos + nv * dt


@wp.kernel
def _proj_loss(p: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec2), valid: float,
               Rc: wp.mat33, tc: wp.vec3, fx: float, fy: float, cx: float, cy: float,
               scale: float, loss: wp.array(dtype=float)):
    pc = Rc * p[0] + tc
    z = wp.max(pc[2], 1.0e-6)
    du = fx * pc[0] / z + cx - target[0][0]
    dv = fy * pc[1] / z + cy - target[0][1]
    wp.atomic_add(loss, 0, scale * valid * (du * du + dv * dv))


class DiffSphere:
    def __init__(self, p0, radius, ground_z, n_frames, substeps=25, fps=24.0,
                 v0=(0.0, 0.0, 0.0), k=5000.0, cd=8.0, mu=0.4, g=-9.81, requires_grad=True):
        rg = requires_grad
        self.substeps, self.n_frames = substeps, n_frames
        self.n_steps = n_frames * substeps
        self.dt = 1.0 / (fps * substeps)
        self.contact_z = float(ground_z + radius)     # centre height at contact
        self.k, self.g = float(k), float(g)
        self.p0 = np.asarray([p0], np.float32)
        self.v0 = wp.array([wp.vec3(*[float(x) for x in v0])], dtype=wp.vec3, requires_grad=rg)
        self.cd = wp.array([float(cd)], dtype=float, requires_grad=rg)
        self.mu = wp.array([float(mu)], dtype=float, requires_grad=rg)
        mk = lambda: [wp.zeros(1, dtype=wp.vec3, requires_grad=rg) for _ in range(self.n_steps + 1)]
        self.pos, self.vel = mk(), mk()
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_v0(self, v): self.v0.assign(np.array([v], np.float32))
    def set_cd(self, c): self.cd.assign(np.array([float(c)], np.float32))
    def set_mu(self, m): self.mu.assign(np.array([float(m)], np.float32))

    def rollout(self):
        self.pos[0].assign(self.p0)
        wp.launch(_set_v0, 1, inputs=[self.v0], outputs=[self.vel[0]])
        for t in range(self.n_steps):
            wp.launch(_step, 1,
                      inputs=[self.pos[t], self.vel[t], self.contact_z, self.k,
                              self.cd, self.mu, self.g, self.dt],
                      outputs=[self.pos[t + 1], self.vel[t + 1]])
    # cd, mu are wp.arrays so their adjoints flow back for the fit

    def frame_pos(self):
        return np.stack([self.pos[t * self.substeps].numpy()[0] for t in range(self.n_frames + 1)])
