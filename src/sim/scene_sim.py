"""M6: real multi-asset rigid scene. N scanned objects on a static surface
(real table mesh or a ground plane), each with a recoverable per-object physics
theta. Object-object contact couples their masses, so relative density becomes
observable (unlike the single-object drop where it is a pure gauge).

Per-object theta = [log_density, mu, restitution, v0x, v0y, v0z] (6). Stacked to
a flat vector of length 6*N. Contact material + mass/inertia mutated in place.

Observation = per-object 8 bbox-corner marker world positions per frame,
concatenated. Recovery uses the FD-Jacobian LM (XPBD gradients are zero through
contact — established in M5).
"""

import numpy as np
import warp as wp

import newton

from ..data.assets import decimate, load_asset

PER = 6  # theta params per object
SUBNAMES = ["log_density", "mu", "restitution", "v0x", "v0y", "v0z"]


def theta_vec_from_cfg(obj_list: list) -> np.ndarray:
    out = []
    for o in obj_list:
        out += [np.log(float(o["density"])), float(o["mu"]), float(o["restitution"]),
                *[float(x) for x in o["v0"]]]
    return np.array(out, dtype=np.float64)


def theta_natural(v: np.ndarray, n_obj: int) -> list:
    out = []
    for i in range(n_obj):
        s = v[i * PER:(i + 1) * PER]
        out.append({"density": float(np.exp(s[0])), "mu": float(s[1]), "restitution": float(s[2]),
                    "v0": [round(float(x), 4) for x in s[3:6]]})
    return out


class SceneSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg["sim"]
        self.fps, self.num_frames, self.substeps = int(s["fps"]), int(s["num_frames"]), int(s["substeps"])
        self.sim_dt = 1.0 / (self.fps * self.substeps)
        self.n_obj = len(cfg["objects"])

        builder = newton.ModelBuilder()
        # --- static environment ---
        self.table_top_z = 0.0
        if cfg["environment"] == "table":
            # Collision proxy: a static box for the tabletop (robust contact).
            # The real table mesh is used only for RENDERING (scripts/render_scene.py),
            # positioned so its top aligns with this box — a standard visual-mesh /
            # collision-proxy split (object-vs-static-triangle-mesh contact is
            # unreliable here; a box is exact).
            t = cfg["table"]
            self.table_top_z = float(t["top_z"])
            hx, hy, hz = float(t["half_x"]), float(t["half_y"]), float(t["thickness"]) / 2
            top = builder.add_body(is_kinematic=True,
                                   xform=wp.transform(wp.vec3(0.0, 0.0, self.table_top_z - hz), wp.quat_identity()))
            builder.add_shape_box(top, hx=hx, hy=hy, hz=hz,
                                  cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
        else:
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))

        # --- dynamic objects ---
        self.markers_local, self.obj_shape, self.base_mass, self.base_inertia = [], [], [], []
        self.mesh_verts, self.mesh_faces, self.start_pos, self.obj_body = [], [], [], []
        self.base_density = 1000.0
        for o in cfg["objects"]:
            tm = decimate(load_asset("rigid", o["name"]), int(o["target_faces"]))
            verts = tm.vertices.astype(np.float64)
            com = verts.mean(0)
            lo, hi = verts.min(0) - com, verts.max(0) - com
            self.markers_local.append(np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
            self.mesh_verts.append((verts - com))
            self.mesh_faces.append(tm.faces.astype(np.int32))
            pos = [float(x) for x in o["pos"]]
            self.start_pos.append(pos)
            body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
            mesh = newton.Mesh((verts - com).astype(np.float32), tm.faces.ravel().astype(np.int32))
            # SDF collision: triangle-mesh vs triangle-mesh contact between two
            # scanned objects is unstable; an SDF gives robust volumetric contact.
            mesh.build_sdf(max_resolution=int(cfg.get("sdf_resolution", 64)))
            builder.add_shape_mesh(body, mesh=mesh,
                                   cfg=newton.ModelBuilder.ShapeConfig(density=self.base_density, mu=0.5, restitution=0.3))
            self.obj_shape.append(builder.shape_count - 1)
            self.obj_body.append(body)

        self.model = builder.finalize()
        bm = self.model.body_mass.numpy()
        bi = self.model.body_inertia.numpy()
        self.base_mass = [float(bm[self.obj_body[i]]) for i in range(self.n_obj)]
        self.base_inertia = [bi[self.obj_body[i]].copy() for i in range(self.n_obj)]
        self.init_body_q = self.model.state().body_q.numpy().copy()  # preserves kinematic tabletop pose

        self.solver = newton.solvers.SolverXPBD(
            self.model, iterations=int(s["xpbd_iterations"]), enable_restitution=True)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1 = self.model.state(), self.model.state()

    def set_theta(self, theta: np.ndarray):
        mass = self.model.body_mass.numpy().copy()
        inv_mass = self.model.body_inv_mass.numpy().copy()
        inertia = self.model.body_inertia.numpy().copy()
        inv_inertia = self.model.body_inv_inertia.numpy().copy()
        mu = self.model.shape_material_mu.numpy().copy()
        rest = self.model.shape_material_restitution.numpy().copy()
        for i in range(self.n_obj):
            s = theta[i * PER:(i + 1) * PER]
            bi = self.obj_body[i]
            density = float(np.clip(np.exp(s[0]), 1.0, 1.0e5))
            sc = density / self.base_density
            m = self.base_mass[i] * sc
            mass[bi] = m; inv_mass[bi] = 1.0 / m
            I = self.base_inertia[i] * sc
            inertia[bi] = I; inv_inertia[bi] = np.linalg.inv(I)
            mu[self.obj_shape[i]] = max(0.0, s[1])
            rest[self.obj_shape[i]] = float(np.clip(s[2], 0.0, 0.99))
        self.model.body_mass.assign(mass); self.model.body_inv_mass.assign(inv_mass)
        self.model.body_inertia.assign(inertia); self.model.body_inv_inertia.assign(inv_inertia)
        self.model.shape_material_mu.assign(mu); self.model.shape_material_restitution.assign(rest)
        self._theta = theta.copy()

    def _reset(self):
        q = self.init_body_q.copy()  # keeps the kinematic tabletop pose
        qd = np.zeros((self.model.body_count, 6), dtype=np.float32)
        for i in range(self.n_obj):
            bi = self.obj_body[i]
            q[bi, :3] = self.start_pos[i]; q[bi, 3:] = [0, 0, 0, 1]  # identity quat
            qd[bi, :3] = self._theta[i * PER + 3:i * PER + 6]  # linear v0
        self.state0.body_q.assign(q); self.state0.body_qd.assign(qd)

    def rollout_poses(self) -> np.ndarray:
        """Returns per-object poses (F+1, N, 7) — object bodies only."""
        self._reset()
        idx = self.obj_body
        poses = [self.state0.body_q.numpy()[idx].copy()]
        a, b = self.state0, self.state1
        sub = 0
        for _ in range(self.num_frames * self.substeps):
            a.clear_forces()
            self.pipeline.collide(a, self.contacts)
            self.solver.step(a, b, None, self.contacts, self.sim_dt)
            a, b = b, a
            sub += 1
            if sub % self.substeps == 0:
                poses.append(a.body_q.numpy()[idx].copy())
        return np.array(poses)

    def rollout(self) -> np.ndarray:
        """Returns markers (F+1, N*8, 3)."""
        poses = self.rollout_poses()
        F = len(poses)
        out = np.empty((F, self.n_obj * 8, 3))
        for i in range(self.n_obj):
            ml = self.markers_local[i]
            for t in range(F):
                p, q = poses[t, i, :3], poses[t, i, 3:]
                out[t, i * 8:(i + 1) * 8] = p + _rotate(q, ml)
        return out


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T
