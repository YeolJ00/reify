"""6-DOF differentiable rigid-body collision with friction, on real mesh assets.

Extends the translation-only M9 contact to full rigid dynamics:
  * orientation (quaternion) + angular velocity, integrated with Euler's equation
    (world inertia I_w = R I_body R^T, gyroscopic term included);
  * contact forces produce TORQUE via the contact-point arm r x f;
  * regularized Coulomb FRICTION f_t = -mu |f_n| v_t / (|v_t| + eps), a smooth
    (differentiable) approximation that opposes tangential slip at the contact.

Momentum conserved by construction: the contact force pair (f on A, -f on B) acts
at the SAME contact point, so both linear and angular momentum are preserved.
Everything is a smooth Warp kernel -> the trajectory-loss gradient flows to
density (and would flow to friction/restitution too). Real geometry via the
sphere-covering from M9.
"""

import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset
from .diff_collide_mesh import sphere_cover

V_EPS = 1.0e-3   # friction regularization (m/s)


def unit_mass_inertia(centers):
    """Per-unit-mass inertia tensor of a uniform point cloud (mat33)."""
    c = centers.astype(np.float64)
    G = np.zeros((3, 3))
    for r in c:
        G += (r @ r) * np.eye(3) - np.outer(r, r)
    G /= len(c)
    G += np.eye(3) * 1e-6  # regularize
    return G


@wp.kernel
def mass_inertia(log_density: wp.array(dtype=float), volume: wp.array(dtype=float),
                 mass: wp.array(dtype=float), inv_mass: wp.array(dtype=float)):
    i = wp.tid()
    m = wp.exp(log_density[i]) * volume[i]
    mass[i] = m
    inv_mass[i] = 1.0 / m


@wp.kernel
def contact_6dof(
    n1: int, ca: wp.array(dtype=wp.vec3), cb: wp.array(dtype=wp.vec3), ra: float, rb: float,
    pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
    vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
    k: float,
    cd_arr: wp.array(dtype=float),          # normal damping (sets restitution) — recoverable
    mu_arr: wp.array(dtype=float),          # Coulomb friction — recoverable
    force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    cd = cd_arr[0]
    mu = mu_arr[0]
    a = i / n1
    b = i % n1
    oa = wp.quat_rotate(rot[0], ca[a])       # world sphere offset (rotated)
    ob = wp.quat_rotate(rot[1], cb[b])
    pa = pos[0] + oa
    pb = pos[1] + ob
    d = pa - pb
    dist = wp.length(d)
    overlap = ra + rb - dist
    if overlap > 0.0 and dist > 1.0e-9:
        n = d / dist
        cpt = 0.5 * (pa + pb)                # contact point (shared -> momentum conserved)
        r0 = cpt - pos[0]
        r1 = cpt - pos[1]
        vc0 = vlin[0] + wp.cross(vang[0], r0)   # velocity of the contact point on each body
        vc1 = vlin[1] + wp.cross(vang[1], r1)
        vrel = vc0 - vc1
        vn = wp.dot(vrel, n)
        fn = wp.max(k * overlap - cd * vn, 0.0)  # normal penalty (no adhesion)
        vt = vrel - vn * n                     # tangential slip
        vt_mag = wp.length(vt)
        ft = -mu * fn * vt / (vt_mag + V_EPS)  # regularized Coulomb friction
        f = fn * n + ft
        wp.atomic_add(force, 0, f)
        wp.atomic_add(torque, 0, wp.cross(r0, f))
        wp.atomic_add(force, 1, -f)
        wp.atomic_add(torque, 1, wp.cross(r1, -f))


@wp.kernel
def integrate_6dof(
    pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
    vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float), inv_mass: wp.array(dtype=float),
    G: wp.array(dtype=wp.mat33), Ginv: wp.array(dtype=wp.mat33),
    gravity: wp.vec3, dt: float,
    pos_o: wp.array(dtype=wp.vec3), rot_o: wp.array(dtype=wp.quat),
    vlin_o: wp.array(dtype=wp.vec3), vang_o: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    # linear
    vl = vlin[i] + (force[i] * inv_mass[i] + gravity) * dt
    vlin_o[i] = vl
    pos_o[i] = pos[i] + vl * dt
    # angular: world inertia from body inertia (I_body = mass * G)
    R = wp.quat_to_matrix(rot[i])
    Rt = wp.transpose(R)
    Iw = R * (mass[i] * G[i]) * Rt
    Iw_inv = R * (inv_mass[i] * Ginv[i]) * Rt
    gyro = wp.cross(vang[i], Iw * vang[i])            # gyroscopic term
    va = vang[i] + (Iw_inv * (torque[i] - gyro)) * dt
    vang_o[i] = va
    # orientation: q_dot = 0.5 * omega_quat * q ; integrate + renormalize
    wq = wp.quat(va[0], va[1], va[2], 0.0)
    qdot = wq * rot[i] * 0.5
    rot_o[i] = wp.normalize(rot[i] + qdot * dt)


@wp.kernel
def accum_pos_loss(pos: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3),
                   scale: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    d = pos[i] - target[i]
    wp.atomic_add(loss, 0, scale * wp.dot(d, d))


@wp.kernel
def accum_quat_loss(rot: wp.array(dtype=wp.quat), target: wp.array(dtype=wp.quat),
                    scale: float, loss: wp.array(dtype=float)):
    # orientation mismatch 1 - (q . q_target)^2  (0 when aligned) -> makes friction observable
    i = wp.tid()
    dp = wp.dot(rot[i], target[i])
    wp.atomic_add(loss, 0, scale * (1.0 - dp * dp))


class DiffCollide6DOF:
    def __init__(self, names, pos0, vel0, ang0=None, pitch=0.012, dt=2.0e-4, n_steps=1600,
                 k=3000.0, cd=5.0, mu=0.5, gravity=(0, 0, 0), requires_grad=True):
        assert len(names) == 2
        self.n, self.dt, self.n_steps = 2, dt, n_steps
        self.k = float(k)
        self.gravity = wp.vec3(*gravity)
        rg = requires_grad
        # friction (mu) and normal damping (cd, sets restitution) are recoverable params
        self.mu = wp.array([float(mu)], dtype=float, requires_grad=rg)
        self.cd = wp.array([float(cd)], dtype=float, requires_grad=rg)

        covers, vols, rads, Gs, Ginvs = [], [], [], [], []
        for name in names:
            tm = decimate(load_asset("rigid", name), 400)
            centers, r = sphere_cover(tm, pitch)
            covers.append(centers); rads.append(r)
            vols.append(float(abs(tm.volume)) if tm.is_watertight else float(len(centers)) * pitch ** 3)
            G = unit_mass_inertia(centers)
            Gs.append(G.astype(np.float32)); Ginvs.append(np.linalg.inv(G).astype(np.float32))
        self.n_spheres = [len(c) for c in covers]
        self.ca = wp.array(covers[0], dtype=wp.vec3)
        self.cb = wp.array(covers[1], dtype=wp.vec3)
        self.ra, self.rb = rads
        self.n_pairs = self.n_spheres[0] * self.n_spheres[1]
        self.volume = wp.array(np.array(vols, np.float32), dtype=float)
        self.G = wp.array(np.stack(Gs), dtype=wp.mat33)
        self.Ginv = wp.array(np.stack(Ginvs), dtype=wp.mat33)

        self.pos0 = np.asarray(pos0, np.float32)
        self.vlin0 = np.asarray(vel0, np.float32)
        self.vang0 = np.asarray(ang0 if ang0 is not None else np.zeros((2, 3)), np.float32)
        self.log_density = wp.zeros(2, dtype=float, requires_grad=rg)
        self.mass = wp.zeros(2, dtype=float, requires_grad=rg)
        self.inv_mass = wp.zeros(2, dtype=float, requires_grad=rg)
        mk = lambda dt_: [wp.zeros(2, dtype=dt_, requires_grad=rg) for _ in range(n_steps + 1)]
        self.pos, self.rot = mk(wp.vec3), mk(wp.quat)
        self.vlin, self.vang = mk(wp.vec3), mk(wp.vec3)
        self.force, self.torque = mk(wp.vec3), mk(wp.vec3)
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)

    def set_log_density(self, logd):
        self.log_density.assign(np.asarray(logd, np.float32))

    def set_friction(self, mu):
        self.mu.assign(np.array([float(mu)], np.float32))

    def set_damping(self, cd):
        self.cd.assign(np.array([float(cd)], np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.rot[0].assign(np.tile([0, 0, 0, 1], (2, 1)).astype(np.float32))
        self.vlin[0].assign(self.vlin0)
        self.vang[0].assign(self.vang0)
        wp.launch(mass_inertia, 2, inputs=[self.log_density, self.volume],
                  outputs=[self.mass, self.inv_mass])
        for t in range(self.n_steps):
            self.force[t].zero_(); self.torque[t].zero_()
            wp.launch(contact_6dof, self.n_pairs,
                      inputs=[self.n_spheres[1], self.ca, self.cb, self.ra, self.rb,
                              self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.k, self.cd, self.mu],
                      outputs=[self.force[t], self.torque[t]])
            # note: self.cd, self.mu are differentiable arrays -> their gradients flow
            wp.launch(integrate_6dof, 2,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.force[t], self.torque[t], self.mass, self.inv_mass,
                              self.G, self.Ginv, self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1], self.vang[t + 1]])

    def positions(self, stride=1):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def orientations(self, stride=1):
        return np.stack([self.rot[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def linear_momentum(self):
        m = self.mass.numpy()
        return np.stack([(self.vlin[t].numpy() * m[:, None]).sum(0) for t in range(self.n_steps + 1)])

    def angular_momentum(self):
        """Total angular momentum about the origin: sum r_com x (m v) + I_w omega."""
        m = self.mass.numpy(); G = self.G.numpy()
        L = []
        for t in range(self.n_steps + 1):
            p = self.pos[t].numpy(); vl = self.vlin[t].numpy()
            q = self.rot[t].numpy(); wang = self.vang[t].numpy()
            tot = np.zeros(3)
            for i in range(2):
                R = _quat_mat(q[i])
                Iw = R @ (m[i] * G[i]) @ R.T
                tot += np.cross(p[i], m[i] * vl[i]) + Iw @ wang[i]
            L.append(tot)
        return np.stack(L)


def _quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
