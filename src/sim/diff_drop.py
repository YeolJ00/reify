"""Differentiable single-object drop onto a ground/table plane, in Warp.

A real scanned object (sphere-covered, as in M9) with 6-DOF rigid dynamics falls
under gravity, strikes a horizontal plane, and bounces/slides. Contact is the same
differentiable penalty+friction as the collision engine, now against a fixed plane:
so restitution (normal damping cd) and friction (mu) are recoverable by gradient
descent. Density is a true gauge here (free fall + Coulomb contact are mass-
independent), so we recover the material parameters, not mass.
"""
import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset
from .diff_collide_6dof import _quat_mat, unit_mass_inertia
from .diff_collide_mesh import sphere_cover

V_EPS = 1.0e-3


@wp.kernel
def ground_contact(
    centers: wp.array(dtype=wp.vec3), radius: float, ground_z: float,
    pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
    vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
    k: float, cd_arr: wp.array(dtype=float), mu_arr: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    cd = cd_arr[0]
    mu = mu_arr[0]
    off = wp.quat_rotate(rot[0], centers[i])
    world = pos[0] + off
    pen = (ground_z + radius) - world[2]
    if pen > 0.0:
        up = wp.vec3(0.0, 0.0, 1.0)
        cp = wp.vec3(world[0], world[1], ground_z)      # contact point on the plane
        r = cp - pos[0]
        vc = vlin[0] + wp.cross(vang[0], r)             # contact-point velocity
        vn = vc[2]
        fn = wp.max(k * pen - cd * vn, 0.0)
        vt = vc - vn * up
        vt_mag = wp.length(vt)
        ft = -mu * fn * vt / (vt_mag + V_EPS)
        f = fn * up + ft
        wp.atomic_add(force, 0, f)
        wp.atomic_add(torque, 0, wp.cross(r, f))


@wp.kernel
def integrate_6dof_1(
    pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
    vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
    mass: float, G: wp.mat33, Ginv: wp.mat33, gravity: wp.vec3, dt: float,
    pos_o: wp.array(dtype=wp.vec3), rot_o: wp.array(dtype=wp.quat),
    vlin_o: wp.array(dtype=wp.vec3), vang_o: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    vl = vlin[i] + (force[i] / mass + gravity) * dt
    vlin_o[i] = vl
    pos_o[i] = pos[i] + vl * dt
    R = wp.quat_to_matrix(rot[i])
    Rt = wp.transpose(R)
    Iw = R * (mass * G) * Rt
    Iwi = R * ((1.0 / mass) * Ginv) * Rt
    gyro = wp.cross(vang[i], Iw * vang[i])
    va = vang[i] + (Iwi * (torque[i] - gyro)) * dt
    vang_o[i] = va
    wq = wp.quat(va[0], va[1], va[2], 0.0)
    rot_o[i] = wp.normalize(rot[i] + (wq * rot[i] * 0.5) * dt)


class DiffDrop:
    def __init__(self, name, pos0, vel0, ang0, density=800.0, ground_z=0.0,
                 pitch=0.012, dt=2.0e-4, n_steps=2000, k=4000.0, cd=8.0, mu=0.5,
                 gravity=(0, 0, -9.81), requires_grad=True):
        self.dt, self.n_steps, self.k, self.ground_z = dt, n_steps, float(k), float(ground_z)
        self.gravity = wp.vec3(*gravity)
        rg = requires_grad
        tm = decimate(load_asset("rigid", name), 400)
        centers, r = sphere_cover(tm, pitch)
        self.centers = wp.array(centers, dtype=wp.vec3)
        self.radius = r
        vol = float(abs(tm.volume)) if tm.is_watertight else float(len(centers)) * pitch ** 3
        self.mass = float(density) * vol
        G = unit_mass_inertia(centers)
        self.G = wp.mat33(*G.astype(np.float32).ravel())
        self.Ginv = wp.mat33(*np.linalg.inv(G).astype(np.float32).ravel())
        self.mesh_V = (tm.vertices - tm.vertices.mean(0)).astype(np.float64)
        self.mesh_F = tm.faces.astype(np.int32)

        self.pos0 = np.asarray([pos0], np.float32)
        self.vlin0 = np.asarray([vel0], np.float32)
        self.vang0 = np.asarray([ang0], np.float32)
        self.mu = wp.array([float(mu)], dtype=float, requires_grad=rg)
        self.cd = wp.array([float(cd)], dtype=float, requires_grad=rg)
        mk = lambda d: [wp.zeros(1, dtype=d, requires_grad=rg) for _ in range(n_steps + 1)]
        self.pos, self.rot = mk(wp.vec3), mk(wp.quat)
        self.vlin, self.vang = mk(wp.vec3), mk(wp.vec3)
        self.force, self.torque = mk(wp.vec3), mk(wp.vec3)
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_friction(self, mu): self.mu.assign(np.array([float(mu)], np.float32))
    def set_damping(self, cd): self.cd.assign(np.array([float(cd)], np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.rot[0].assign(np.array([[0, 0, 0, 1]], np.float32))
        self.vlin[0].assign(self.vlin0)
        self.vang[0].assign(self.vang0)
        for t in range(self.n_steps):
            self.force[t].zero_(); self.torque[t].zero_()
            wp.launch(ground_contact, len(self.centers.numpy()),
                      inputs=[self.centers, self.radius, self.ground_z, self.pos[t], self.rot[t],
                              self.vlin[t], self.vang[t], self.k, self.cd, self.mu],
                      outputs=[self.force[t], self.torque[t]])
            wp.launch(integrate_6dof_1, 1,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.force[t], self.torque[t], self.mass, self.G, self.Ginv,
                              self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1], self.vang[t + 1]])

    def positions(self, stride=1):
        return np.stack([self.pos[t].numpy()[0] for t in range(0, self.n_steps + 1, stride)])

    def orientations(self, stride=1):
        return np.stack([self.rot[t].numpy()[0] for t in range(0, self.n_steps + 1, stride)])
