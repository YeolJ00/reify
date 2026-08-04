"""One scene, many excitations — the 'physics lab' rig.

Two real scanned assets resting on / moving over a table, with 6-DOF rigid dynamics,
differentiable-style penalty contact against the ground AND between the objects, plus
Coulomb friction. The SAME scene is driven by different initial conditions ("probes"):

    drop     — object A falls onto the table            -> excites restitution
    slide    — object A is pushed across the table      -> excites friction
    collide  — object A is pushed into object B         -> excites the mass ratio

The recoverable parameters are the contact material (cd = normal damping, i.e.
restitution; mu = friction) and the two objects' DENSITY RATIO — which is a gauge
under gravity+contact alone and only becomes observable when the objects collide.

Forward rollout only (recovery is done by grid+CEM, which we've shown tolerates the
rugged contact landscape better than gradients). (warp env.)
"""
import os

import numpy as np
import warp as wp

# Hunt-Crossley damping (force proportional to penetration x velocity) is the DEFAULT.
# Linear damping (PROBE_HUNT_CROSSLEY=0) is kept only for reproducing older results: its
# restitution has a spurious velocity dependence whose sign flips with the damping value
# (cd=5 -> -0.055, cd=20 -> +0.058), because clamping `max(k*pen - cd*v, 0)` at separation
# behaves differently depending on how damping-dominated the contact is. Hunt-Crossley
# vanishes at separation without a clamp and gives de/dv <= 0 throughout, which is the
# direction real materials show.
HC = float(os.environ.get("PROBE_HUNT_CROSSLEY", "1"))

from ..data.assets import decimate, load_asset
from .diff_collide_6dof import integrate_6dof, unit_mass_inertia
from .diff_collide_mesh import sphere_cover

V_EPS = 1.0e-3


@wp.kernel
def world_spheres(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                  center_local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                  world: wp.array(dtype=wp.vec3)):
    s = wp.tid()
    b = body[s]
    world[s] = pos[b] + wp.quat_rotate(rot[b], center_local[s])


@wp.kernel
def ground_contact_n(world: wp.array(dtype=wp.vec3), radius: float, body: wp.array(dtype=int),
                     pos: wp.array(dtype=wp.vec3), vlin: wp.array(dtype=wp.vec3),
                     vang: wp.array(dtype=wp.vec3), ground_z: float, k: float,
                     cd: wp.array(dtype=float), mu: wp.array(dtype=float),
                     force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3)):
    s = wp.tid()
    b = body[s]
    pen = (ground_z + radius) - world[s][2]
    if pen > 0.0:
        cpt = wp.vec3(world[s][0], world[s][1], ground_z)
        r = cpt - pos[b]
        vc = vlin[b] + wp.cross(vang[b], r)
        # Hunt-Crossley: damping scales with penetration. Linear damping (cd*v) has to be
        # clamped at zero to avoid adhesion, and that clamp is what gave our contact a
        # restitution that RISES with impact speed (+0.058 per m/s measured on a control)
        # where a spring-damper should be velocity-independent and real materials fall.
        fn = wp.max(k * pen - HC * cd[0] * pen * vc[2] - (1.0 - HC) * cd[0] * vc[2], 0.0)
        vt = wp.vec3(vc[0], vc[1], 0.0)
        ft = -mu[0] * fn * vt / (wp.length(vt) + V_EPS)
        f = wp.vec3(ft[0], ft[1], fn)
        wp.atomic_add(force, b, f)
        wp.atomic_add(torque, b, wp.cross(r, f))


@wp.kernel
def pair_contact_n(nS: int, world: wp.array(dtype=wp.vec3), radius: float,
                   body: wp.array(dtype=int), pos: wp.array(dtype=wp.vec3),
                   vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
                   k: float, cd: wp.array(dtype=float), mu: wp.array(dtype=float),
                   force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    i = t / nS
    j = t % nS
    if i >= j:
        return
    bi = body[i]
    bj = body[j]
    if bi == bj:
        return
    d = world[i] - world[j]
    dist = wp.length(d)
    overlap = 2.0 * radius - dist
    if overlap > 0.0 and dist > 1.0e-9:
        n = d / dist
        cpt = 0.5 * (world[i] + world[j])          # shared contact point -> momentum conserved
        ri = cpt - pos[bi]
        rj = cpt - pos[bj]
        vrel = (vlin[bi] + wp.cross(vang[bi], ri)) - (vlin[bj] + wp.cross(vang[bj], rj))
        vn = wp.dot(vrel, n)
        fn = wp.max(k * overlap - HC * cd[0] * overlap * vn
                    - (1.0 - HC) * cd[0] * vn, 0.0)
        vt = vrel - vn * n
        ft = -mu[0] * fn * vt / (wp.length(vt) + V_EPS)
        f = fn * n + ft
        wp.atomic_add(force, bi, f)
        wp.atomic_add(torque, bi, wp.cross(ri, f))
        wp.atomic_add(force, bj, -f)
        wp.atomic_add(torque, bj, wp.cross(rj, -f))


class ProbeScene:
    def __init__(self, names, pos0, vel0, ang0=None, densities=(600.0, 600.0), ground_z=0.706,
                 pitch=0.020, dt=2.0e-4, n_steps=1400, k=4000.0, cd=8.0, mu=0.5,
                 gravity=(0.0, 0.0, -9.81), ball_radius=None, mesh_scale=None):
        # ball_radius: model each body as a single solid sphere instead of a scanned mesh
        # (used for the generated-video probes, where the objects really are balls)
        self.N = len(names); self.dt, self.n_steps = dt, n_steps
        # CONTACT STABILITY. The contact is an explicit penalty spring-damper, so the
        # damping impulse must not be able to reverse a body's velocity within one step:
        #   cd < 2m/dt
        # Beyond that the integrator pumps energy instead of removing it. Measured on a
        # 3.7 cm sphere (m=0.127 kg, dt=0.69 ms, bound 367): cd=200 gives e=0.00, cd=600
        # gives e=5.2 and cd=2000 gives e=9.9 -- a ball leaving the ground 98x higher than
        # it was dropped from, which no passive contact can do. A parameter search that
        # ranges past this bound is searching a region where the answer is an artefact.
        self._cd_limit = None
        self.k, self.ground_z = float(k), float(ground_z)
        self.gravity = wp.vec3(*gravity)

        cl, body, vols = [], [], []
        if ball_radius is not None:
            R = float(ball_radius)
            self.radius = R
            for bi in range(self.N):
                cl.append(np.zeros((1, 3), np.float32)); body.append(np.array([bi], np.int32))
                vols.append(4.0 / 3.0 * np.pi * R ** 3)
        else:
            for bi, name in enumerate(names):
                cat, _, nm = name.rpartition("/")
                tm = decimate(load_asset(cat or "rigid", nm), 400)
                if mesh_scale is not None:
                    sfac = float(mesh_scale[bi] if hasattr(mesh_scale, "__len__") else mesh_scale)
                    tm = tm.copy(); tm.apply_scale(sfac)
                    pitch_b = pitch * sfac      # keep the cover resolution proportional
                else:
                    pitch_b = pitch
                centers, r = sphere_cover(tm, pitch_b)
                self.radius = float(r)
                cl.append(centers); body.append(np.full(len(centers), bi, np.int32))
                vols.append(float(abs(tm.volume)) if tm.is_watertight else len(centers) * pitch_b ** 3)
        self.center_local = wp.array(np.concatenate(cl), dtype=wp.vec3)
        self.body = wp.array(np.concatenate(body), dtype=int)
        self.nS = int(self.body.shape[0])
        self.volumes = np.asarray(vols, np.float64)
        self._ball_R = ball_radius

        self.mu = wp.array([float(mu)], dtype=float)
        self.cd = wp.array([float(cd)], dtype=float)
        self.mass = wp.zeros(self.N, dtype=float); self.inv_mass = wp.zeros(self.N, dtype=float)
        self.G = wp.zeros(self.N, dtype=wp.mat33); self.Ginv = wp.zeros(self.N, dtype=wp.mat33)
        Gs, Gis = [], []
        for bi in range(self.N):
            if ball_radius is not None:
                G = (2.0 / 5.0) * float(ball_radius) ** 2 * np.eye(3)   # solid sphere
            else:
                G = unit_mass_inertia(np.concatenate(cl)[np.concatenate(body) == bi])
            Gs.append(G.astype(np.float32)); Gis.append(np.linalg.inv(G).astype(np.float32))
        self.G.assign(np.stack(Gs)); self.Ginv.assign(np.stack(Gis))
        self.set_densities(densities)

        self.pos0 = np.asarray(pos0, np.float32)
        self.vlin0 = np.asarray(vel0, np.float32)
        self.vang0 = np.asarray(ang0 if ang0 is not None else np.zeros((self.N, 3)), np.float32)
        mk = lambda d: [wp.zeros(self.N, dtype=d) for _ in range(n_steps + 1)]
        m_min = float(np.min(self.mass.numpy())) if hasattr(self, "mass") else None
        if m_min:
            # half the hard bound: at 0.9x (cd=330 here) the model was still misbehaving,
            # giving e=0.64 where cd=200 gives e=0.00. Restitution is monotonic in cd only
            # well below the limit, which is the region a parameter search can use.
            # Under Hunt-Crossley the damping force is cd*penetration*v, so the same
            # bound applies to the PRODUCT: cd*pen < 2m/dt. A typical penetration is
            # mg/k, so the admissible cd is larger by that factor. Without this the
            # linear bound (183 here) clamped every Hunt-Crossley value to the same
            # number and the comparison silently measured nothing.
            base = 0.5 * (2.0 * m_min / self.dt)
            if HC > 0.0:
                pen_ref = max(m_min * 9.81 / max(float(k), 1e-9), 1e-6)
                base = base / pen_ref
            self._cd_limit = base
            if cd > self._cd_limit:
                import warnings
                warnings.warn(
                    f"cd={cd:.1f} exceeds the explicit-integration stability bound "
                    f"2m/dt={self._cd_limit:.0f}; contact will create energy. Clamping.",
                    RuntimeWarning)
                cd = self._cd_limit * 0.9
                self.cd = wp.array([float(cd)], dtype=float)   # stays a warp array
        self.pos, self.rot = mk(wp.vec3), mk(wp.quat)
        self.vlin, self.vang = mk(wp.vec3), mk(wp.vec3)
        self.force, self.torque = wp.zeros(self.N, dtype=wp.vec3), wp.zeros(self.N, dtype=wp.vec3)
        self.world = wp.zeros(self.nS, dtype=wp.vec3)

    def set_densities(self, dens):
        m = np.asarray(dens, np.float64) * self.volumes
        self.mass.assign(m.astype(np.float32)); self.inv_mass.assign((1.0 / m).astype(np.float32))

    def set_mu(self, v): self.mu.assign(np.array([float(v)], np.float32))
    def set_cd(self, v): self.cd.assign(np.array([float(v)], np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.rot[0].assign(np.tile([0, 0, 0, 1.0], (self.N, 1)).astype(np.float32))
        self.vlin[0].assign(self.vlin0)
        self.vang[0].assign(self.vang0)
        for t in range(self.n_steps):
            self.force.zero_(); self.torque.zero_()
            wp.launch(world_spheres, self.nS,
                      inputs=[self.pos[t], self.rot[t], self.center_local, self.body], outputs=[self.world])
            wp.launch(ground_contact_n, self.nS,
                      inputs=[self.world, self.radius, self.body, self.pos[t], self.vlin[t], self.vang[t],
                              self.ground_z, self.k, self.cd, self.mu], outputs=[self.force, self.torque])
            if self.N > 1:
                wp.launch(pair_contact_n, self.nS * self.nS,
                          inputs=[self.nS, self.world, self.radius, self.body, self.pos[t],
                                  self.vlin[t], self.vang[t], self.k, self.cd, self.mu],
                          outputs=[self.force, self.torque])
            wp.launch(integrate_6dof, self.N,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t], self.force, self.torque,
                              self.mass, self.inv_mass, self.G, self.Ginv, self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1], self.vang[t + 1]])

    def positions(self, stride=20):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def rotations(self, stride=20):
        """Per-frame body orientations as quaternions (x, y, z, w).

        Needed to RENDER the simulation rather than composite it: a falling vase tumbles,
        and a position-only playback cannot show that.
        """
        return np.stack([self.rot[t].numpy() for t in range(0, self.n_steps + 1, stride)])
