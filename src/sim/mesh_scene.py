"""Rigid-body scene whose contact geometry IS the mesh (no sphere cover).

Drop-in for ProbeScene: same constructor arguments, same positions()/rotations() output,
same per-body mu/cd/mass and table material, same gravity_seq ramp. Only the collision
representation changes.
"""
import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset
from .diff_collide_6dof import integrate_6dof
from .mesh_contact import (apply_revolute, ground_contact_mesh, pair_contact_mesh,
                           vertex_weights, world_verts)


class MeshProbeScene:
    def __init__(self, names, pos0, vel0, ang0=None, rot0=None, densities=None, masses=None,
                 ground_z=0.706, dt=1.0 / 1440.0, n_steps=1400, k=2500.0, cd=3000.0,
                 mu=0.5, gravity=(0.0, 0.0, -9.81), mesh_scale=None, faces=1200,
                 ground_mu=None, ground_cd=None, query_dist=0.05, table=None):
        self.N = len(names)
        self.dt, self.n_steps = float(dt), int(n_steps)
        self.ground_z = float(ground_z)
        self.gravity = wp.vec3(*gravity)
        self.gravity_seq = None
        self._gmu = float(ground_mu) if ground_mu is not None else -1.0
        self._gcd = float(ground_cd) if ground_cd is not None else -1.0
        self.query_dist = float(query_dist)
        # (half_width, half_depth) of the table, or None for an unbounded plane
        self.half_w, self.half_d = (float(table[0]), float(table[1])) if table else (-1.0, -1.0)

        loc, bod, wts, self.meshes, self.coms, self.sizes, vols = [], [], [], [], [], [], []
        mesh_ids = []
        for bi, name in enumerate(names):
            if hasattr(name, "vertices"):
                # a trimesh passed directly -- procedural rig parts (balance beam, pans)
                # live in the same scene as scanned assets and need the same exact contact
                tm = name.copy()
            else:
                cat, _, nm = name.rpartition("/")
                tm = decimate(load_asset(cat or "rigid", nm), faces).copy()
            if mesh_scale is not None:
                s = float(mesh_scale[bi] if hasattr(mesh_scale, "__len__") else mesh_scale)
                tm.apply_scale(s)
            V = np.asarray(tm.vertices, np.float64)
            C = V.mean(0)
            # Body frame is COM-centred, matching how poses are exported. Unlike the sphere
            # cover this is the ONLY offset -- the collision surface and the visible surface
            # are now the same object, so nothing floats.
            Vc = (V - C).astype(np.float32)
            self.coms.append(C)
            self.sizes.append(float(max(tm.extents)))
            vols.append(float(abs(tm.volume)) if tm.is_watertight else float(np.prod(tm.extents)) * 0.5)
            loc.append(Vc)
            bod.append(np.full(len(Vc), bi, np.int32))
            wts.append(vertex_weights(tm).astype(np.float32))
            m = wp.Mesh(points=wp.array(Vc, dtype=wp.vec3),
                        indices=wp.array(np.asarray(tm.faces, np.int32).flatten(), dtype=int))
            self.meshes.append(m); mesh_ids.append(m.id)

        self.local = wp.array(np.concatenate(loc), dtype=wp.vec3)
        self.body = wp.array(np.concatenate(bod), dtype=int)
        self.weight = wp.array(np.concatenate(wts), dtype=float)
        self.mesh_ids = wp.array(np.array(mesh_ids, np.uint64), dtype=wp.uint64)
        self.nV = len(self.body)
        self.volumes = np.asarray(vols)

        self.mass = wp.zeros(self.N, dtype=float)
        self.inv_mass = wp.zeros(self.N, dtype=float)
        self.G = wp.zeros(self.N, dtype=wp.mat33)
        self.Ginv = wp.zeros(self.N, dtype=wp.mat33)
        self._loc_np, self._bod_np, self._w_np = (np.concatenate(loc), np.concatenate(bod),
                                                  np.concatenate(wts))
        self.set_masses(masses if masses is not None
                        else np.asarray(densities) * self.volumes)

        arr = lambda v: wp.array(np.full(self.N, float(v), np.float32) if np.isscalar(v)
                                 else np.asarray(v, np.float32).reshape(self.N), dtype=float)
        self.mu, self.cd, self.k = arr(mu), arr(cd), arr(k)

        self.pos0 = np.asarray(pos0, np.float32)
        self.vlin0 = np.asarray(vel0, np.float32)
        self.vang0 = np.asarray(ang0 if ang0 is not None else np.zeros((self.N, 3)), np.float32)
        # INITIAL ORIENTATION. rot[0] was hard-coded to identity, so a body could only ever
        # start in its authored pose -- which makes a rocking probe impossible, since a bowl
        # sitting flat does not rock. Supplying rot0 lets the excitation be a controlled
        # DISPLACEMENT (tip it and release) rather than a velocity kick, and a displacement
        # is what makes the initial amplitude the same for every candidate.
        self.rot0 = (np.tile([0.0, 0.0, 0.0, 1.0], (self.N, 1)).astype(np.float32)
                     if rot0 is None else np.asarray(rot0, np.float32).reshape(self.N, 4))
        mk = lambda d: [wp.zeros(self.N, dtype=d) for _ in range(self.n_steps + 1)]
        self.pos, self.rot = mk(wp.vec3), mk(wp.quat)
        self.vlin, self.vang = mk(wp.vec3), mk(wp.vec3)
        self.force = wp.zeros(self.N, dtype=wp.vec3)
        self.torque = wp.zeros(self.N, dtype=wp.vec3)
        self.world = wp.zeros(self.nV, dtype=wp.vec3)
        # revolute joints: off for every body unless set_hinge() is called
        self.is_hinge = wp.zeros(self.N, dtype=int)
        self.anchor = wp.zeros(self.N, dtype=wp.vec3)
        self.axis = wp.array(np.tile([0.0, 1.0, 0.0], (self.N, 1)).astype(np.float32),
                             dtype=wp.vec3)

    def set_masses(self, m):
        m = np.asarray(m, np.float64).reshape(self.N)
        self.mass.assign(m.astype(np.float32))
        self.inv_mass.assign((1.0 / m).astype(np.float32))
        # Inertia from the area-weighted vertex cloud -- the same distribution the contact
        # forces act on, so torque and inertia stay consistent even for open meshes.
        Gs, Gis = [], []
        for b in range(self.N):
            P = self._loc_np[self._bod_np == b]
            w = self._w_np[self._bod_np == b]; w = w / w.sum()
            r2 = (P ** 2 * w[:, None]).sum(0)
            G = np.diag([r2[1] + r2[2], r2[0] + r2[2], r2[0] + r2[1]])
            off = np.zeros((3, 3))
            for a in range(3):
                for c in range(3):
                    if a != c:
                        off[a, c] = -(P[:, a] * P[:, c] * w).sum()
            G = (G + off) * m[b]
            G += np.eye(3) * (1e-9 + 1e-6 * np.trace(G) / 3.0)   # keep it invertible
            Gs.append(G.astype(np.float32)); Gis.append(np.linalg.inv(G).astype(np.float32))
        self.G.assign(np.stack(Gs)); self.Ginv.assign(np.stack(Gis))

    def set_hinge(self, bi, anchor, axis=(0.0, 1.0, 0.0), damp_per_sec=6.0):
        """Pin body `bi` to `anchor`, free to rotate only about `axis`.

        `damp_per_sec` is the exponential decay rate of angular velocity, converted to a
        per-step factor -- so the settling time is a physical time, independent of dt.
        """
        self._hinge_damp = float(np.exp(-float(damp_per_sec) * self.dt))
        return self._set_hinge_arrays(bi, anchor, axis)

    def _set_hinge_arrays(self, bi, anchor, axis=(0.0, 1.0, 0.0)):
        """Pin body `bi` to `anchor`, free to rotate only about `axis`. Exact, no stiffness."""
        h = self.is_hinge.numpy(); h[bi] = 1; self.is_hinge.assign(h)
        a = self.anchor.numpy(); a[bi] = np.asarray(anchor, np.float32); self.anchor.assign(a)
        x = self.axis.numpy(); x[bi] = np.asarray(axis, np.float32); self.axis.assign(x)

    def calibrate_stiffness(self, pen_frac=0.005):
        """k_i so each body sinks pen_frac of its own size under its own weight.

        Forces are area-weighted and sum to 1 per body, so unlike the sphere cover there is
        no contact-count factor to measure -- only the fraction of the surface actually
        touching, which the weights already account for.
        """
        m = self.mass.numpy(); sz = np.asarray(self.sizes)
        self.k.assign((m * 9.81 / (pen_frac * sz)).astype(np.float32))

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.rot[0].assign(self.rot0)
        self.vlin[0].assign(self.vlin0)
        self.vang[0].assign(self.vang0)
        gseq = self.gravity_seq
        for t in range(self.n_steps):
            g_t = self.gravity if gseq is None else wp.vec3(*gseq[t])
            self.force.zero_(); self.torque.zero_()
            wp.launch(world_verts, self.nV,
                      inputs=[self.pos[t], self.rot[t], self.local, self.body],
                      outputs=[self.world])
            wp.launch(ground_contact_mesh, self.nV,
                      inputs=[self.world, self.body, self.pos[t], self.vlin[t], self.vang[t],
                              self.ground_z, self.k, self.cd, self.mu, self._gmu, self._gcd,
                              self.weight, self.half_w, self.half_d],
                      outputs=[self.force, self.torque])
            if self.N > 1:
                wp.launch(pair_contact_mesh, self.nV * self.N,
                          inputs=[self.world, self.body, self.mesh_ids, self.N,
                                  self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                                  self.k, self.cd, self.mu, self.weight, self.query_dist],
                          outputs=[self.force, self.torque])
            wp.launch(integrate_6dof, self.N,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.force, self.torque, self.mass, self.inv_mass,
                              self.G, self.Ginv, g_t, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1],
                               self.vlin[t + 1], self.vang[t + 1]])
            wp.launch(apply_revolute, self.N,
                      inputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1],
                              self.vang[t + 1], self.is_hinge, self.anchor, self.axis,
                              getattr(self, "_hinge_damp", 1.0)])

    def rest_height(self, bi):
        """Table-relative z that puts this body's lowest vertex exactly on the table."""
        P = self._loc_np[self._bod_np == bi]
        return self.ground_z - float(P[:, 2].min())

    def positions(self, stride=60):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def rotations(self, stride=60):
        return np.stack([self.rot[t].numpy() for t in range(0, self.n_steps + 1, stride)])
