"""Contact against the ACTUAL mesh, replacing the sphere-cover approximation.

Why this exists
---------------
The old contact voxelised each mesh into interior spheres of one common radius and then
never touched the mesh again. Measured against sampled surface points, that collision
surface was wrong by 2.28 mm on average and up to 10.6 mm at edges -- and the error is
not just cosmetic:

  * a flat object's contact surface became bumpy, so slip angle picked up a geometric
    contribution that is not friction;
  * "contact area" was a voxel-resolution artifact (1 to 458 spheres) rather than a
    physical footprint, and the per-body stiffness calibration keyed off it;
  * the collision surface sat below the visible surface, so objects rendered floating.

If the geometry is approximated by spheres there is little reason to run a simulator on
authored meshes at all -- the mesh IS the asset.

Two contact paths
-----------------
**Ground.** The table is a plane, and a linear function over a triangle attains its
minimum at a vertex, so testing every mesh VERTEX against the plane is *exact* -- no
queries, no approximation, no radius.

**Body-body.** Vertices of body i are queried against the mesh of body j using Warp's
BVH (`mesh_query_point_sign_normal`). The query runs in body j's LOCAL frame -- the world
vertex is pulled back through j's transform -- so each wp.Mesh is built once at setup and
never rebuilt or refitted as bodies move. That is what makes this affordable per step.

Both paths are plain penalty contact with Hunt-Crossley damping and regularised Coulomb
friction, matching the previous kernels term for term so results stay comparable.
"""
import numpy as np
import warp as wp

V_EPS = 1.0e-3          # friction regularisation, m/s
HC = 1.0                # Hunt-Crossley damping (force ~ penetration * velocity)


@wp.kernel
def world_verts(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    b = body[i]
    out[i] = pos[b] + wp.quat_rotate(rot[b], local[i])


@wp.kernel
def ground_contact_mesh(world: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                        pos: wp.array(dtype=wp.vec3), vlin: wp.array(dtype=wp.vec3),
                        vang: wp.array(dtype=wp.vec3), ground_z: float,
                        k: wp.array(dtype=float), cd: wp.array(dtype=float),
                        mu: wp.array(dtype=float), ground_mu: float, ground_cd: float,
                        weight: wp.array(dtype=float), half_w: float, half_d: float,
                        force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3)):
    """Exact plane contact: every mesh vertex tested directly against the table."""
    i = wp.tid()
    b = body[i]
    # A BOUNDED table. The ground was an infinite plane, so nothing could ever fall off an
    # edge -- which makes "does it leave the table" unrepresentable, and that is exactly the
    # threshold form the bounce probe needs. Negative bounds keep the old infinite plane.
    if half_w > 0.0 and (wp.abs(world[i][0]) > half_w or wp.abs(world[i][1]) > half_d):
        return
    pen = ground_z - world[i][2]
    if pen > 0.0:
        r = world[i] - pos[b]
        vc = vlin[b] + wp.cross(vang[b], r)
        cde = cd[b]
        mue = mu[b]
        if ground_cd > 0.0:
            cde = wp.sqrt(cd[b] * ground_cd)
        if ground_mu > 0.0:
            mue = wp.sqrt(mu[b] * ground_mu)
        # weight[i] is the vertex's share of the surface (its Voronoi area, normalised).
        # Without it a densely-tesselated region would push harder than a sparse one
        # purely because it has more vertices -- the same defect as counting spheres.
        w = weight[i]
        fn = wp.max(k[b] * w * pen - HC * cde * w * pen * vc[2], 0.0)
        vt = wp.vec3(vc[0], vc[1], 0.0)
        ft = -mue * fn * vt / (wp.length(vt) + V_EPS)
        f = wp.vec3(ft[0], ft[1], fn)
        wp.atomic_add(force, b, f)
        wp.atomic_add(torque, b, wp.cross(r, f))


@wp.kernel
def pair_contact_mesh(world: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                      meshes: wp.array(dtype=wp.uint64), N: int,
                      pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                      vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
                      k: wp.array(dtype=float), cd: wp.array(dtype=float),
                      mu: wp.array(dtype=float), weight: wp.array(dtype=float),
                      max_dist: float,
                      force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3)):
    """Vertices of one body against the true surface of another, via Warp's mesh BVH.

    The query is done in the OTHER body's local frame, so its wp.Mesh is built once and
    never refitted -- moving bodies cost nothing extra.
    """
    t = wp.tid()
    i = t / N
    j = t % N
    bi = body[i]
    if bi == j:
        return
    # pull the world vertex back into body j's local frame
    inv = wp.quat_inverse(rot[j])
    q = wp.quat_rotate(inv, world[i] - pos[j])
    res = wp.mesh_query_point_sign_normal(meshes[j], q, max_dist)
    if not res.result:
        return
    if res.sign > 0.0:          # outside body j -- no contact
        return
    cp_local = wp.mesh_eval_position(meshes[j], res.face, res.u, res.v)
    d = wp.length(q - cp_local)
    if d < 1.0e-9:
        return
    pen = d
    n_local = (q - cp_local) / d
    n = wp.quat_rotate(rot[j], -n_local)        # points from j into i
    cpt = pos[j] + wp.quat_rotate(rot[j], cp_local)
    ri = cpt - pos[bi]
    rj = cpt - pos[j]
    vrel = (vlin[bi] + wp.cross(vang[bi], ri)) - (vlin[j] + wp.cross(vang[j], rj))
    vn = wp.dot(vrel, n)
    kij = k[bi] * k[j] / (k[bi] + k[j])         # springs in series
    cde = wp.sqrt(cd[bi] * cd[j])
    mue = wp.sqrt(mu[bi] * mu[j])
    w = weight[i]
    fn = wp.max(kij * w * pen - HC * cde * w * pen * vn, 0.0)
    vt = vrel - vn * n
    ft = -mue * fn * vt / (wp.length(vt) + V_EPS)
    f = fn * n + ft
    wp.atomic_add(force, bi, f)
    wp.atomic_add(torque, bi, wp.cross(ri, f))
    wp.atomic_add(force, j, -f)
    wp.atomic_add(torque, j, wp.cross(rj, -f))


def vertex_weights(mesh):
    """Each vertex's share of the surface area, normalised to sum to 1 per body.

    Penalty force is applied per vertex, so an unweighted sum makes finely-tesselated
    regions stiffer than coarse ones for no physical reason. Weighting by the vertex's
    one-third share of its incident triangles makes total force independent of how the
    mesh happens to be triangulated.
    """
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    a = np.zeros(len(V))
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    for c in range(3):
        np.add.at(a, F[:, c], area / 3.0)
    s = a.sum()
    return (a / s if s > 0 else np.full(len(V), 1.0 / max(len(V), 1)))


@wp.kernel
def apply_revolute(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                   vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
                   is_hinge: wp.array(dtype=int), anchor: wp.array(dtype=wp.vec3),
                   axis: wp.array(dtype=wp.vec3), damp: wp.array(dtype=float),
                   limit: wp.array(dtype=float)):
    """Reduced-coordinate revolute joint, applied as a projection after integration.

    A pan balance needs a real pivot, and a penalty spring would introduce a compliance whose
    stiffness has to be tuned and which biases the tipping threshold -- exactly the kind of
    knob that turned the tilt detector's offset into a calibration step. Projecting the state
    onto the joint's allowed DOF instead is EXACT: the beam cannot translate at all, and can
    only spin about its axis, with no stiffness parameter anywhere.

    Position: pinned to the anchor.
    Rotation: the component about `axis` is kept, everything else is discarded by
              re-forming the quaternion from the axis and the extracted swing angle.
    Velocity: linear zeroed, angular projected onto the axis.
    """
    b = wp.tid()
    if is_hinge[b] == 0:
        return
    a = wp.normalize(axis[b])
    pos[b] = anchor[b]
    vlin[b] = wp.vec3(0.0, 0.0, 0.0)
    # Pivot damping. An ideal frictionless hinge never settles -- the beam swings as an
    # undamped pendulum, so reading the tilt at any fixed frame samples an arbitrary phase
    # of the oscillation rather than the equilibrium (measured 3/7 correct). Every real
    # balance has pivot friction and air drag; without them the probe has no steady state
    # to read at all.
    vang[b] = a * wp.dot(vang[b], a) * damp[b]
    # swing-twist: keep only the twist about `a`
    q = rot[b]
    v = wp.vec3(q[0], q[1], q[2])
    proj = a * wp.dot(v, a)
    tw = wp.quat(proj[0], proj[1], proj[2], q[3])
    n = wp.sqrt(tw[0] * tw[0] + tw[1] * tw[1] + tw[2] * tw[2] + tw[3] * tw[3])
    if n > 1.0e-9:
        tw = wp.quat(tw[0] / n, tw[1] / n, tw[2] / n, tw[3] / n)
    else:
        tw = wp.quat(0.0, 0.0, 0.0, 1.0)
    # ANGLE STOPS. A real balance swings a few degrees, not ninety: its beam meets stops that
    # keep the pans near level. Without them the pans tip with the beam and whatever is in
    # them slides out over the rim -- measured, both weights ended on the TABLE, so the
    # reading came from the moment before they escaped rather than from a weighing.
    if limit[b] > 0.0:
        ang = 2.0 * wp.atan2(wp.dot(wp.vec3(tw[0], tw[1], tw[2]), a), tw[3])
        if ang > limit[b] or ang < -limit[b]:
            cl = wp.clamp(ang, -limit[b], limit[b])
            hs = wp.sin(cl * 0.5)
            tw = wp.quat(a[0] * hs, a[1] * hs, a[2] * hs, wp.cos(cl * 0.5))
            vang[b] = wp.vec3(0.0, 0.0, 0.0)      # dead stop, no bounce
    rot[b] = tw


@wp.kernel
def apply_links(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
                vlin: wp.array(dtype=wp.vec3), vang: wp.array(dtype=wp.vec3),
                link_a: wp.array(dtype=int), link_b: wp.array(dtype=int),
                local_a: wp.array(dtype=wp.vec3), local_b: wp.array(dtype=wp.vec3),
                stiff: wp.array(dtype=float), damp: wp.array(dtype=float),
                force: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3)):
    """Point-to-point spring linking two bodies -- the suspension a hanging pan needs.

    Why a spring and not a projection: `apply_revolute` pins a body to a FIXED world anchor,
    which cannot express "hangs from a point that is itself moving". Snapping the pan onto the
    beam's end each step would express it, but a hard projection teleports the pan without the
    beam ever feeling its weight -- and the load is the entire measurement. A penalty link
    exchanges equal and opposite forces, so the beam is loaded by what its pans carry.

    The attachment point sits ABOVE each pan's centre of mass, so gravity levels the pan on
    its own. That is how a real balance keeps its pans flat through the swing, and it removes
    the need for the angle stops that were previously holding the contents in.
    """
    i = wp.tid()
    a = link_a[i]
    b = link_b[i]
    if a < 0 or b < 0:
        return
    pa = pos[a] + wp.quat_rotate(rot[a], local_a[i])
    pb = pos[b] + wp.quat_rotate(rot[b], local_b[i])
    d = pa - pb
    ra = pa - pos[a]
    rb = pb - pos[b]
    va = vlin[a] + wp.cross(vang[a], ra)
    vb = vlin[b] + wp.cross(vang[b], rb)
    f = -stiff[i] * d - damp[i] * (va - vb)
    wp.atomic_add(force, a, f)
    wp.atomic_add(torque, a, wp.cross(ra, f))
    wp.atomic_add(force, b, -f)
    wp.atomic_add(torque, b, wp.cross(rb, -f))
