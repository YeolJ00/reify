"""Differentiable, momentum-conserving penalty contact in pure Warp.

The fix for the M7b root cause. Instead of Newton's XPBD contact (non-
differentiable, non-momentum-conserving), we write contact as a custom Warp
kernel — exactly the pattern we already use for the wind force. Two guarantees
hold by construction:

  * momentum conservation: the pairwise force formula is antisymmetric
    (F_ij = -F_ji), so sum of contact forces = 0 -> total momentum conserved
    exactly (verified to machine precision).
  * differentiability: Warp auto-generates the adjoint of every kernel, so the
    gradient of a trajectory loss flows back through the whole rollout to the
    parameters (density) — the thing XPBD/VBD could not give.

Bodies are spheres (analytic penetration); mass = density * (4/3 pi r^3), so
density is the recoverable parameter. Cost: stiff penalty needs small dt (slow,
but exact and differentiable — the accepted trade). Extends to meshes by
swapping the sphere-overlap term for an SDF penetration query.
"""

import numpy as np
import warp as wp

PI = 3.14159265358979


@wp.kernel
def masses_from_logdensity(log_density: wp.array(dtype=float), radius: wp.array(dtype=float),
                           mass: wp.array(dtype=float), inv_mass: wp.array(dtype=float)):
    i = wp.tid()
    r = radius[i]
    vol = (4.0 / 3.0) * PI * r * r * r
    m = wp.exp(log_density[i]) * vol
    mass[i] = m
    inv_mass[i] = 1.0 / m


@wp.kernel
def contact_and_gravity_forces(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    n: int,
    k: float,          # penalty stiffness
    c: float,          # normal damping (sets restitution)
    gravity: wp.vec3,
    ground_z: float,
    force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    f = gravity * mass[i]   # gravity (external; does not affect the pair balance)

    # pairwise penalty contact — antisymmetric => total momentum conserved
    for j in range(n):
        if j != i:
            d = pos[i] - pos[j]
            dist = wp.length(d)
            overlap = radius[i] + radius[j] - dist
            if overlap > 0.0:
                nrm = d / (dist + 1.0e-9)
                vrel = wp.dot(vel[i] - vel[j], nrm)
                fmag = k * overlap - c * vrel
                f = f + fmag * nrm

    # ground penalty (external contact with a fixed plane at ground_z)
    pen = ground_z + radius[i] - pos[i][2]
    if pen > 0.0:
        f = f + wp.vec3(0.0, 0.0, k * pen - c * vel[i][2])

    force[i] = f


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
              force: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float), dt: float,
              pos_out: wp.array(dtype=wp.vec3), vel_out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = vel[i] + force[i] * inv_mass[i] * dt   # semi-implicit Euler
    vel_out[i] = v
    pos_out[i] = pos[i] + v * dt


@wp.kernel
def accum_pos_loss(pos: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3),
                   scale: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    d = pos[i] - target[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class DiffCollide:
    """Differentiable N-sphere collision rollout in Warp."""

    def __init__(self, radius, pos0, vel0, dt=1.0e-3, n_steps=600, k=5000.0, c=5.0,
                 gravity=(0.0, 0.0, 0.0), ground_z=None, requires_grad=True):
        self.n = len(radius)
        self.dt = dt
        self.n_steps = n_steps
        self.k = float(k)
        self.c = float(c)
        self.gravity = wp.vec3(*gravity)
        self.ground_z = float(ground_z) if ground_z is not None else -1.0e9
        rg = requires_grad

        self.radius = wp.array(np.asarray(radius, np.float32), dtype=float)
        self.pos0 = np.asarray(pos0, np.float32)
        self.vel0 = np.asarray(vel0, np.float32)
        self.log_density = wp.zeros(self.n, dtype=float, requires_grad=rg)
        self.mass = wp.zeros(self.n, dtype=float, requires_grad=rg)
        self.inv_mass = wp.zeros(self.n, dtype=float, requires_grad=rg)

        self.pos = [wp.zeros(self.n, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.vel = [wp.zeros(self.n, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.force = [wp.zeros(self.n, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_log_density(self, logd):
        self.log_density.assign(np.asarray(logd, np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.vel[0].assign(self.vel0)
        wp.launch(masses_from_logdensity, self.n, inputs=[self.log_density, self.radius],
                  outputs=[self.mass, self.inv_mass])
        for t in range(self.n_steps):
            wp.launch(contact_and_gravity_forces, self.n,
                      inputs=[self.pos[t], self.vel[t], self.radius, self.mass, self.n,
                              self.k, self.c, self.gravity, self.ground_z],
                      outputs=[self.force[t]])
            wp.launch(integrate, self.n,
                      inputs=[self.pos[t], self.vel[t], self.force[t], self.inv_mass, self.dt],
                      outputs=[self.pos[t + 1], self.vel[t + 1]])

    def positions(self, stride=1):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def velocities(self, stride=1):
        return np.stack([self.vel[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def total_momentum(self):
        m = self.mass.numpy()
        return np.stack([(self.vel[t].numpy() * m[:, None]).sum(0)
                         for t in range(self.n_steps + 1)])
