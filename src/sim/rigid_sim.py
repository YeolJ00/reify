"""M5: real-asset rigid-body drop sim with a recoverable physical theta.

theta = [log_density, mu, restitution, v0(3), w0(3)]  (9 params). density sets
mass + inertia (both scale linearly with density for fixed geometry); mu and
restitution are the object's contact material; v0/w0 are the initial spatial
velocity. All are mutated in place on the finalized model (no rebuild per eval).

Observation = world positions of the mesh's 8 bounding-box corners transformed
by the body pose at each frame (a rigid-body analog of the cloth vertex
trajectory; captures both translation and orientation). XPBD + CollisionPipeline
provide contact. Gradients through XPBD contact are not expected to flow (checked
in scripts/check_grad_rigid.py) -> recovery uses the solver-agnostic FD-Jacobian
LM in src/optimize/lm.py.

Verified against newton 1.4.0 installed API.
"""

import numpy as np
import warp as wp

import newton

from ..data.assets import decimate, load_asset

THETA_NAMES = ["log_density", "mu", "restitution", "v0x", "v0y", "v0z", "w0x", "w0y", "w0z"]
THETA_DIM = 9


def theta_vec_from_cfg(tc: dict) -> np.ndarray:
    return np.array(
        [np.log(float(tc["density"])), float(tc["mu"]), float(tc["restitution"]),
         *[float(x) for x in tc["v0"]], *[float(x) for x in tc["w0"]]],
        dtype=np.float64,
    )


def theta_natural(v: np.ndarray) -> dict:
    return {"density": float(np.exp(v[0])), "mu": float(v[1]), "restitution": float(v[2]),
            "v0": [round(float(x), 4) for x in v[3:6]], "w0": [round(float(x), 4) for x in v[6:9]]}


class RigidDropSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg["sim"]
        self.fps = int(s["fps"])
        self.num_frames = int(s["num_frames"])
        self.substeps = int(s["substeps"])
        self.sim_dt = 1.0 / (self.fps * self.substeps)

        a = cfg["asset"]
        mesh_tm = decimate(load_asset(a["category"], a["name"]), int(a["target_faces"]))
        verts = mesh_tm.vertices.astype(np.float32)
        # markers: 8 bbox corners (in body/local frame, COM-centered)
        com = verts.mean(axis=0)
        lo, hi = verts.min(0) - com, verts.max(0) - com
        self.markers_local = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
            dtype=np.float64,
        )
        self.drop_z = float(a["drop_height"])

        builder = newton.ModelBuilder()
        gcfg = newton.ModelBuilder.ShapeConfig(mu=float(s["ground_mu"]))
        builder.add_ground_plane(cfg=gcfg)
        # place body COM at drop height; center mesh on COM via shape xform
        self.body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, self.drop_z), wp.quat_identity()))
        self.mesh_verts = (verts - com).astype(np.float64)  # body-frame, COM-centered
        self.mesh_faces = mesh_tm.faces.astype(np.int32)
        mesh = newton.Mesh((verts - com).astype(np.float32), mesh_tm.faces.ravel().astype(np.int32))
        self.base_density = 1000.0
        ocfg = newton.ModelBuilder.ShapeConfig(density=self.base_density, mu=0.5, restitution=0.3)
        builder.add_shape_mesh(self.body, mesh=mesh, cfg=ocfg)

        self.model = builder.finalize()
        self.obj_shape = self.model.shape_count - 1  # last shape added
        # base mass/inertia at base_density, for linear density scaling
        self.base_mass = float(self.model.body_mass.numpy()[0])
        self.base_inertia = self.model.body_inertia.numpy()[0].copy()

        self.solver = newton.solvers.SolverXPBD(
            self.model, iterations=int(s["xpbd_iterations"]), enable_restitution=True)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state0 = self.model.state()
        self.state1 = self.model.state()

    def set_theta(self, theta: np.ndarray):
        # density is a near-gauge direction (see M5 findings): LM drifts it freely,
        # so clamp to a physical range to avoid mass->0 (division by zero).
        density = float(np.clip(np.exp(theta[0]), 1.0, 1.0e5))
        scale = density / self.base_density
        mass = self.base_mass * scale
        self.model.body_mass.assign(np.array([mass], dtype=np.float32))
        self.model.body_inv_mass.assign(np.array([1.0 / mass], dtype=np.float32))
        self.model.body_inertia.assign((self.base_inertia * scale)[None])
        self.model.body_inv_inertia.assign(np.linalg.inv(self.base_inertia * scale)[None].astype(np.float32))
        # object contact material (index obj_shape); leave ground (index 0) fixed
        mu = self.model.shape_material_mu.numpy(); mu[self.obj_shape] = max(0.0, theta[1])
        self.model.shape_material_mu.assign(mu)
        e = self.model.shape_material_restitution.numpy()
        e[self.obj_shape] = float(np.clip(theta[2], 0.0, 0.99))
        self.model.shape_material_restitution.assign(e)
        self._theta = theta.copy()

    def _reset_state0(self, theta):
        self.state0.body_q.assign(
            np.array([[0.0, 0.0, self.drop_z, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        # newton body_qd is (linear v, angular w) — verified empirically
        qd = np.array([[theta[3], theta[4], theta[5], theta[6], theta[7], theta[8]]], dtype=np.float32)
        self.state0.body_qd.assign(qd)

    def rollout(self) -> np.ndarray:
        """Run the drop. Returns marker world positions (F+1, 8, 3)."""
        return self._markers_from_poses(self.rollout_poses())

    def rollout_poses(self) -> np.ndarray:
        """Run the drop. Returns body poses (F+1, 7) [pos xyz, quat xyzw]."""
        self._reset_state0(self._theta)
        poses = [self.state0.body_q.numpy()[0].copy()]
        sub = 0
        a, b = self.state0, self.state1
        for _ in range(self.num_frames * self.substeps):
            a.clear_forces()
            self.pipeline.collide(a, self.contacts)
            self.solver.step(a, b, None, self.contacts, self.sim_dt)
            a, b = b, a
            sub += 1
            if sub % self.substeps == 0:
                poses.append(a.body_q.numpy()[0].copy())
        return np.array(poses)

    def _markers_from_poses(self, poses: np.ndarray) -> np.ndarray:
        """poses (F+1, 7) [px,py,pz, qx,qy,qz,qw] -> markers (F+1, 8, 3)."""
        out = np.empty((len(poses), len(self.markers_local), 3))
        for t, T in enumerate(poses):
            p, q = T[:3], T[3:]
            out[t] = p + _rotate(q, self.markers_local)
        return out


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate rows of v (N,3) by quaternion q [x,y,z,w]."""
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T
