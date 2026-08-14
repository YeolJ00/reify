"""Cloth and soft-body probes on REAL assets, not procedural grids.

The repo already had deformables, but only procedural ones: `FlagSim` builds a cloth GRID and
`DiffSoft` an `add_soft_grid` cube. Neither touches the scanned assets the project is actually
about. Newton exposes `add_cloth_mesh` and `add_soft_mesh`, both density-parameterised and both
accepting real geometry; neither was used anywhere.

Solver is `SolverSemiImplicit` throughout. `SolverVBD` returns exactly-zero gradients in
newton 1.4.0 (recorded in docs/PROJECT_LOG.md), so anything that might later be differentiated
must not be built on it.
"""
import numpy as np
import newton
import trimesh
import warp as wp


@wp.kernel
def _signed_dist(m: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(m, pts[i], 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(m, q.face, q.u, q.v)
        out[i] = q.sign * wp.length(pts[i] - cp)
    else:
        out[i] = 1.0e6


def cloth_from_mesh(asset_mesh, target_faces=1200, scale=1.0):
    """Decimate a scanned cloth asset down to something a cloth solver can integrate.

    The GSO cloth scans are 2-10 MB dense surfaces; feeding one straight to a cloth solver
    gives ~100k particles and a timestep nothing can afford. Decimation is not cosmetic here,
    it is what makes the asset simulable at all.
    """
    tm = asset_mesh.copy()
    tm.apply_scale(scale)
    if len(tm.faces) > target_faces:
        tm = tm.simplify_quadric_decimation(face_count=target_faces)
    tm.remove_unreferenced_vertices()
    return tm


def tets_from_mesh(asset_mesh, pitch, scale=1.0):
    """Voxelise a closed mesh and split each interior voxel into 5 tetrahedra.

    `add_soft_mesh` wants a tet mesh, and nothing in the stack produces one from a scan.
    Voxel->5-tet is the standard decomposition: it is conforming (shared faces match) as long
    as the parity of the split alternates with the voxel's index sum, which is what `flip`
    below does. Without alternating, neighbouring voxels disagree on their shared diagonal and
    the mesh has cracks.
    """
    tm = asset_mesh.copy()
    tm.apply_scale(scale)
    # A regular grid tested for containment, rather than trimesh's voxelized().fill(), which
    # needs scipy (absent from this env). Containment also gives a genuinely SOLID body --
    # surface voxels alone would tetrahedralise a shell, and a shell has no bulk modulus.
    lo, hi = tm.bounds
    gx = [np.arange(lo[a] + pitch * 0.5, hi[a] + pitch * 0.5, pitch) for a in range(3)]
    P = np.stack(np.meshgrid(*gx, indexing="ij"), -1).reshape(-1, 3)
    # trimesh.contains needs rtree (absent). Warp's own mesh SDF is already a dependency and
    # gives the same answer: a point is interior when the signed distance is negative.
    m = wp.Mesh(points=wp.array(np.asarray(tm.vertices, np.float32), dtype=wp.vec3),
                indices=wp.array(np.asarray(tm.faces, np.int32).flatten(), dtype=int))
    sd = wp.zeros(len(P), dtype=float)
    wp.launch(_signed_dist, len(P), inputs=[m.id, wp.array(P.astype(np.float32), dtype=wp.vec3)],
              outputs=[sd])
    inside = sd.numpy() < 0.0
    if inside.sum() < 8:
        raise ValueError(f"only {inside.sum()} interior voxels at pitch={pitch}")
    origin = np.array([g[0] for g in gx]) - pitch * 0.5
    idx = np.rint((P[inside] - origin) / pitch).astype(np.int64)
    verts, vkey = [], {}

    def vid(c):
        t = tuple(int(x) for x in c)
        if t not in vkey:
            vkey[t] = len(verts)
            verts.append(origin + np.array(t, float) * pitch)
        return vkey[t]

    CORNER = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    EVEN = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]
    ODD = [(0, 1, 2, 5), (0, 2, 3, 7), (0, 4, 5, 7), (2, 5, 6, 7), (0, 2, 5, 7)]
    tets = []
    for c in idx:
        corners = [vid(c + d) for d in CORNER]
        flip = int((c[0] + c[1] + c[2]) % 2)
        for t in (ODD if flip else EVEN):
            tets.append([corners[i] for i in t])
    V = np.asarray(verts, np.float64)
    T = np.asarray(tets, np.int32)
    # orient every tet positively; a negative-volume tet gives negative stiffness
    a, b, c2, d = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]], V[T[:, 3]]
    neg = np.einsum("ij,ij->i", np.cross(b - a, c2 - a), d - a) < 0
    T[neg] = T[neg][:, [0, 2, 1, 3]]
    return V, T


def build_cloth_model(tm, density, tri_ke, tri_kd, edge_ke, pos=(0, 0, 1.0), ground_z=0.0):
    b = newton.ModelBuilder()
    b.add_cloth_mesh(pos=wp.vec3(*pos), rot=wp.quat_identity(), scale=1.0,
                     vel=wp.vec3(0.0, 0.0, 0.0),
                     vertices=[wp.vec3(*v) for v in np.asarray(tm.vertices, np.float32)],
                     indices=np.asarray(tm.faces, np.int32).flatten().tolist(),
                     density=float(density), tri_ke=float(tri_ke), tri_ka=float(tri_ke),
                     tri_kd=float(tri_kd), edge_ke=float(edge_ke), edge_kd=1.0e-3)
    b.add_ground_plane()
    return b.finalize()


def build_soft_model(V, T, density, k_mu, k_lambda, k_damp, pos=(0, 0, 1.0)):
    b = newton.ModelBuilder()
    mesh = newton.TetMesh(vertices=[wp.vec3(*v) for v in np.asarray(V, np.float32)],
                          tet_indices=np.asarray(T, np.int32).flatten().tolist())
    b.add_soft_mesh(pos=wp.vec3(*pos), rot=wp.quat_identity(), scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0), mesh=mesh, density=float(density),
                    k_mu=float(k_mu), k_lambda=float(k_lambda), k_damp=float(k_damp))
    b.add_ground_plane()
    return b.finalize()
