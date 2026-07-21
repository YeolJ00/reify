"""Pinhole camera: numpy projection for rendering, warp kernel for the
differentiable path (architecture step 7).

Convention: camera looks down its +Z axis; u = fx * Xc/Zc + cx, v = fy * Yc/Zc + cy
with v measured downward (image coordinates). World is Z-up.
"""

import numpy as np
import warp as wp


@wp.kernel
def project_points_kernel(
    points: wp.array(dtype=wp.vec3),
    R: wp.mat33,  # world -> camera rotation
    t: wp.vec3,  # camera translation: Xc = R @ Xw + t
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_uv: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    pc = R * points[tid] + t
    z = wp.max(pc[2], 1.0e-6)  # clamp instead of branch: keeps kernel differentiable
    out_uv[tid] = wp.vec2(fx * pc[0] / z + cx, fy * pc[1] / z + cy)


@wp.kernel
def project_bary_points_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    track_tri: wp.array(dtype=wp.int32),
    track_bary: wp.array(dtype=wp.vec3),
    R: wp.mat33,
    t: wp.vec3,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_uv: wp.array(dtype=wp.vec2),
):
    """Project barycentric surface points (tracks attached to the cloth)."""
    tid = wp.tid()
    tri = track_tri[tid]
    b = track_bary[tid]
    p = (
        particle_q[tri_indices[tri, 0]] * b[0]
        + particle_q[tri_indices[tri, 1]] * b[1]
        + particle_q[tri_indices[tri, 2]] * b[2]
    )
    pc = R * p + t
    z = wp.max(pc[2], 1.0e-6)
    out_uv[tid] = wp.vec2(fx * pc[0] / z + cx, fy * pc[1] / z + cy)


class Camera:
    """Look-at pinhole camera built from the config `m4.camera` section."""

    def __init__(self, cam_cfg: dict):
        self.width = int(cam_cfg["width"])
        self.height = int(cam_cfg["height"])
        eye = np.asarray(cam_cfg["eye"], dtype=np.float64)
        target = np.asarray(cam_cfg["target"], dtype=np.float64)
        up = np.asarray(cam_cfg.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)

        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)  # image v grows downward

        # rows of R map world into (right, down, forward) camera axes
        self.R = np.stack([right, down, fwd])
        self.t = -self.R @ eye

        fov = np.deg2rad(float(cam_cfg["fov_deg"]))
        self.fx = self.fy = 0.5 * self.width / np.tan(0.5 * fov)
        self.cx = 0.5 * self.width
        self.cy = 0.5 * self.height

    def project(self, pts: np.ndarray):
        """Numpy projection. pts (N,3) -> uv (N,2), depth (N,)."""
        pc = pts @ self.R.T + self.t
        z = np.maximum(pc[:, 2], 1e-6)
        u = self.fx * pc[:, 0] / z + self.cx
        v = self.fy * pc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1), pc[:, 2]

    def wp_args(self):
        """(R, t, fx, fy, cx, cy) in warp types, for the projection kernels."""
        return (
            wp.mat33(*self.R.astype(np.float32).ravel()),
            wp.vec3(*self.t.astype(np.float32)),
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
        )
