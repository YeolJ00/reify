"""Settling / relaxation pass for a roughly-placed multi-object scene.

Takes N rigid assets placed at rough poses (floating, interpenetrating) above a
static support (the tabletop) and relaxes them into a physically consistent, sim-ready
resting state: no interpenetration, each object resting on the table or on another
object. The settled poses are the consistent initial state for the inverse pipeline.

Method: POSITION-BASED relaxation (projected depenetration + a gravity bias), not
explicit penalty dynamics — the latter ejects/jitters when objects start deeply
overlapping. Each iteration pushes overlapping spheres apart (ground + object-object)
and nudges bodies down toward support; it is unconditionally stable and converges
monotonically. Translation is relaxed; input orientations are kept (assets are placed
upright), which is sufficient for a sim-ready initial state. (warp env.)
"""
import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset
from .diff_collide_mesh import sphere_cover


@wp.kernel
def world_spheres(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                  center_local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                  world: wp.array(dtype=wp.vec3)):
    s = wp.tid()
    b = body[s]
    world[s] = pos[b] + wp.quat_rotate(rot[b], center_local[s])


@wp.kernel
def ground_disp(world: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
                body: wp.array(dtype=int), ground_z: float,
                disp: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float)):
    s = wp.tid()
    pen = (ground_z + radius[s]) - world[s][2]
    if pen > 0.0:
        wp.atomic_add(disp, body[s], wp.vec3(0.0, 0.0, pen))     # push straight up out of the floor
        wp.atomic_add(cnt, body[s], 1.0)


@wp.kernel
def pair_disp(nS: int, world: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
              body: wp.array(dtype=int), tol: float, disp: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float)):
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
    # surface-shell spheres of two TOUCHING objects overlap by up to ~a radius; only the
    # overlap BEYOND that tolerance is true interpenetration to be pushed out.
    overlap = (radius[i] + radius[j] - dist) - tol
    if overlap > 0.0 and dist > 1.0e-9:
        push = 0.5 * overlap * (d / dist)                        # split the overlap between the two bodies
        wp.atomic_add(disp, bi, push)
        wp.atomic_add(cnt, bi, 1.0)
        wp.atomic_add(disp, bj, -push)
        wp.atomic_add(cnt, bj, 1.0)


@wp.kernel
def apply_disp(disp: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float),
               relax: float, max_step: float, grav_step: float, pos: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    mv = relax * disp[b]                                          # summed push-out (not averaged)
    n = wp.length(mv)
    if n > max_step:                                             # clamp step -> no overshoot/oscillation
        mv = mv * (max_step / n)
    pos[b] = pos[b] + mv - wp.vec3(0.0, 0.0, grav_step)         # + gravity bias -> seek support


class SettleScene:
    def __init__(self, names, pos0, quat0=None, ground_z=0.706, pitch=0.016,
                 relax=0.25, max_step=3.0e-3, grav_step=6.0e-4, pair_tol=None):
        self.N = len(names); self.ground_z = float(ground_z)
        self.relax = float(relax); self.max_step = float(max_step); self.grav_step = float(grav_step)
        self._pitch = float(pitch)
        self.pair_tol = float(pair_tol) if pair_tol is not None else 0.62 * float(pitch)  # ~one sphere radius
        cl, rad, body = [], [], []
        self.meshes = []
        for bi, name in enumerate(names):
            tm = decimate(load_asset("rigid", name), 400); self.meshes.append(tm)
            centers, r = sphere_cover(tm, pitch)
            cl.append(centers); rad.append(np.full(len(centers), r, np.float32)); body.append(np.full(len(centers), bi, np.int32))
        self.center_local = wp.array(np.concatenate(cl), dtype=wp.vec3)
        self.radius = wp.array(np.concatenate(rad), dtype=float)
        self.body = wp.array(np.concatenate(body), dtype=int)
        self.nS = int(self.body.shape[0])
        q0 = np.tile([0, 0, 0, 1.0], (self.N, 1)) if quat0 is None else np.asarray(quat0, np.float32)
        self.pos = wp.array(np.asarray(pos0, np.float32), dtype=wp.vec3)
        self.rot = wp.array(q0.astype(np.float32), dtype=wp.quat)
        self.disp = wp.zeros(self.N, dtype=wp.vec3); self.cnt = wp.zeros(self.N, dtype=float)
        self.world = wp.zeros(self.nS, dtype=wp.vec3)

    def _world(self):
        wp.launch(world_spheres, self.nS, inputs=[self.pos, self.rot, self.center_local, self.body], outputs=[self.world])

    def step(self, grav_step=None):
        gs = self.grav_step if grav_step is None else grav_step
        self.disp.zero_(); self.cnt.zero_(); self._world()
        wp.launch(ground_disp, self.nS, inputs=[self.world, self.radius, self.body, self.ground_z], outputs=[self.disp, self.cnt])
        wp.launch(pair_disp, self.nS * self.nS, inputs=[self.nS, self.world, self.radius, self.body, self.pair_tol], outputs=[self.disp, self.cnt])
        wp.launch(apply_disp, self.N, inputs=[self.disp, self.cnt, self.relax, self.max_step, gs], outputs=[self.pos])

    def metrics(self):
        """(max penetration mm, max per-iteration motion mm)."""
        self._world()
        w = self.world.numpy(); r = self.radius.numpy(); b = self.body.numpy()
        gpen = np.maximum(0.0, (self.ground_z + r) - w[:, 2]).max()
        ppen = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                wi, wj = w[b == i], w[b == j]; ri, rj = r[b == i][0], r[b == j][0]
                dmat = np.linalg.norm(wi[:, None] - wj[None], axis=2)
                ppen = max(ppen, np.maximum(0.0, (ri + rj - self.pair_tol) - dmat).max())
        return max(gpen, ppen) * 1000.0

    def run(self, n_iters=2200, snapshot_every=30):
        snaps = [(self.pos.numpy().copy(), self.rot.numpy().copy())]
        for t in range(n_iters):
            # gravity bias seeks support, then ramps to 0 over the first 75% so the run
            # ENDS on pure depenetration -> residual overlap driven to ~0
            gs = self.grav_step * max(0.0, 1.0 - t / (0.75 * n_iters))
            self.step(grav_step=gs)
            if t % snapshot_every == 0:
                snaps.append((self.pos.numpy().copy(), self.rot.numpy().copy()))
        snaps.append((self.pos.numpy().copy(), self.rot.numpy().copy()))
        return snaps

    def poses(self):
        return self.pos.numpy().copy(), self.rot.numpy().copy()
