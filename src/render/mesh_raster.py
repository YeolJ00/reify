"""Rasterise a posed mesh through our own camera. No masks, no compositing.

Why this exists: the "simulated video" used to be faked by cutting the object out of a
photograph with a mask and pasting it wherever the rollout put it. Masks are not part of
the physics or the measurement -- they were purely an artefact of that hack, and every
visual defect came from one failing to match the object's shape: a vase whose neck faded
out, a smear where the neck had been, a dark blob where two overlapping instances were cut
out together.

Rendering the actual simulated mesh removes the entire class of problem. There is nothing
to cut out, so there is nothing to cut out wrongly. It is a flat-shaded render rather than
a photoreal one, which is the honest trade: the two panes no longer match in appearance,
but the motion is directly comparable and nothing on screen is fabricated.

Painter's algorithm with per-face Lambert shading is enough here -- the objects are convex
or nearly so, and we are looking at where things are, not at their materials.
"""
import cv2
import numpy as np


def look_at_basis(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float)); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def project(P, eye, fwd, right, upv, fov_deg, w, h):
    """World points -> pixel coords and camera-space depth."""
    d = np.asarray(P, float) - np.asarray(eye, float)
    z = d @ fwd
    x = d @ right
    y = d @ upv
    fx = 0.5 * w / np.tan(np.radians(fov_deg) * 0.5)
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = 0.5 * w + fx * x / zz
    v = 0.5 * h - fx * y / zz
    return np.stack([u, v], 1), z


def render(mesh_v, mesh_f, R, t, cam, bg=None, colour=(232, 232, 236),
           ground_z=None, light=(0.4, -0.6, 0.8)):
    """One frame: mesh at pose (R, t), plus an optional ground plane.

    mesh_v/mesh_f: vertices (N,3) in the body frame, faces (M,3).
    cam: dict with eye, target, fov_deg, width, height.
    """
    w, h = int(cam["width"]), int(cam["height"])
    eye = np.asarray(cam["eye"], float)
    fwd, right, upv = look_at_basis(eye, cam["target"])
    img = (np.zeros((h, w, 3), np.uint8) + np.array([38, 44, 52], np.uint8)
           if bg is None else bg.copy())

    if ground_z is not None:
        quad = np.array([[-1.2, -0.85, ground_z], [0.45, -0.85, ground_z],
                         [0.45, 0.85, ground_z], [-1.2, 0.85, ground_z]])
        uv, z = project(quad, eye, fwd, right, upv, cam["fov_deg"], w, h)
        if (z > 0).all():
            cv2.fillConvexPoly(img, uv.round().astype(np.int32), (128, 96, 66),
                               cv2.LINE_AA)

    V = np.asarray(mesh_v, float) @ np.asarray(R, float).T + np.asarray(t, float)
    uv, z = project(V, eye, fwd, right, upv, cam["fov_deg"], w, h)
    F = np.asarray(mesh_f, int)
    tri_z = z[F].mean(1)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)
    L = np.asarray(light, float); L = L / np.linalg.norm(L)
    shade = np.clip(0.35 + 0.65 * np.abs(n @ L), 0.0, 1.0)
    base = np.asarray(colour, float)
    order = np.argsort(-tri_z)                       # far to near
    for i in order:
        if tri_z[i] <= 0:
            continue
        p = uv[F[i]]
        if not np.isfinite(p).all():
            continue
        c = tuple(int(v) for v in np.clip(base * shade[i], 0, 255))
        cv2.fillConvexPoly(img, p.round().astype(np.int32), c, cv2.LINE_AA)
    return img
