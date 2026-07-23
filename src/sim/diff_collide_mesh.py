"""Differentiable, momentum-conserving contact for REAL MESH assets, in Warp.

Warp's mesh_query (BVH + winding number) conserves momentum but its adjoint is
broken (discrete face selection), so the gradient does not flow. Fix: approximate
each real scanned mesh by a COVERING OF INTERIOR SPHERES (voxelize the mesh), then
use the analytic sphere-sphere penalty contact proven in diff_collide.py. This
keeps the real shape (the sphere set traces the geometry) while every contact term
is a smooth, differentiable function of position — so the trajectory-loss gradient
flows to density, and the equal-and-opposite pair forces conserve momentum exactly.

Translation-only rigid bodies (COM position + linear velocity). Density enters via
mass = density * volume; its gradient flows through the integrator.
"""

import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset


def sphere_cover(mesh, pitch):
    """Voxelize a trimesh into interior sphere centers + a common radius."""
    vox = mesh.voxelized(pitch=pitch)          # surface shell of overlapping spheres
    centers = np.asarray(vox.points, dtype=np.float32)
    centers = centers - mesh.vertices.mean(0)      # COM-centered
    radius = float(pitch) * 0.62                     # slight overlap -> gap-free surface
    return centers, radius


@wp.kernel
def mass_from_logdensity(log_density: wp.array(dtype=float), volume: wp.array(dtype=float),
                         mass: wp.array(dtype=float), inv_mass: wp.array(dtype=float)):
    i = wp.tid()
    m = wp.exp(log_density[i]) * volume[i]
    mass[i] = m
    inv_mass[i] = 1.0 / m


@wp.kernel
def sphere_pair_contact(
    n1: int,                                  # spheres on body 1
    ca: wp.array(dtype=wp.vec3),              # body-0 sphere offsets (local)
    cb: wp.array(dtype=wp.vec3),              # body-1 sphere offsets (local)
    ra: float, rb: float,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    k: float, c: float,
    force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()                              # over all (a,b) sphere pairs
    a = i / n1
    b = i % n1
    pa = pos[0] + ca[a]
    pb = pos[1] + cb[b]
    d = pa - pb
    dist = wp.length(d)
    overlap = ra + rb - dist
    if overlap > 0.0 and dist > 1.0e-9:
        nrm = d / dist
        vrel = wp.dot(vel[0] - vel[1], nrm)
        fmag = k * overlap - c * vrel
        f = fmag * nrm
        wp.atomic_add(force, 0, f)            # push body 0 out
        wp.atomic_add(force, 1, -f)          # equal-and-opposite -> momentum conserved


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
              force: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
              gravity: wp.vec3, dt: float,
              pos_out: wp.array(dtype=wp.vec3), vel_out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = vel[i] + (force[i] * inv_mass[i] + gravity) * dt   # gravity as acceleration
    vel_out[i] = v
    pos_out[i] = pos[i] + v * dt


@wp.kernel
def accum_pos_loss(pos: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3),
                   scale: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    d = pos[i] - target[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


class DiffMeshCollide:
    """Two real scanned meshes (sphere-covered) with differentiable penalty contact."""

    def __init__(self, names, pos0, vel0, pitch=0.014, dt=3.0e-4, n_steps=1400,
                 k=3000.0, c=5.0, gravity=(0, 0, 0), requires_grad=True):
        assert len(names) == 2
        self.n = 2
        self.dt, self.n_steps, self.k, self.c = dt, n_steps, float(k), float(c)
        self.gravity = wp.vec3(*gravity)
        rg = requires_grad

        covers, vols, rads = [], [], []
        for name in names:
            tm = decimate(load_asset("rigid", name), 400)
            centers, r = sphere_cover(tm, pitch)
            covers.append(centers)
            rads.append(r)
            vols.append(float(abs(tm.volume)) if tm.is_watertight
                        else float(len(centers)) * (pitch ** 3))  # voxel-count volume
        self.n_spheres = [len(c) for c in covers]
        self.ca = wp.array(covers[0], dtype=wp.vec3)
        self.cb = wp.array(covers[1], dtype=wp.vec3)
        self.ra, self.rb = rads
        self.n_pairs = self.n_spheres[0] * self.n_spheres[1]

        self.volume = wp.array(np.array(vols, np.float32), dtype=float)
        self.pos0 = np.asarray(pos0, np.float32)
        self.vel0 = np.asarray(vel0, np.float32)
        self.log_density = wp.zeros(2, dtype=float, requires_grad=rg)
        self.mass = wp.zeros(2, dtype=float, requires_grad=rg)
        self.inv_mass = wp.zeros(2, dtype=float, requires_grad=rg)
        self.pos = [wp.zeros(2, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.vel = [wp.zeros(2, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.force = [wp.zeros(2, dtype=wp.vec3, requires_grad=rg) for _ in range(n_steps + 1)]
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_log_density(self, logd):
        self.log_density.assign(np.asarray(logd, np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.vel[0].assign(self.vel0)
        wp.launch(mass_from_logdensity, 2, inputs=[self.log_density, self.volume],
                  outputs=[self.mass, self.inv_mass])
        for t in range(self.n_steps):
            self.force[t].zero_()             # fresh each step before atomic accumulation
            wp.launch(sphere_pair_contact, self.n_pairs,
                      inputs=[self.n_spheres[1], self.ca, self.cb, self.ra, self.rb,
                              self.pos[t], self.vel[t], self.k, self.c],
                      outputs=[self.force[t]])
            wp.launch(integrate, 2,
                      inputs=[self.pos[t], self.vel[t], self.force[t], self.inv_mass,
                              self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.vel[t + 1]])

    def positions(self, stride=1):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def total_momentum(self):
        m = self.mass.numpy()
        return np.stack([(self.vel[t].numpy() * m[:, None]).sum(0) for t in range(self.n_steps + 1)])
