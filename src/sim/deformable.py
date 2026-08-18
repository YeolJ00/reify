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


def cloth_grid_from_asset(asset_mesh, cell=0.012, scale=1.0):
    """A UNIFORM grid sized from the scan -- not the scan's own triangles.

    Decimating the scan and simulating that was wrong twice over.

    Numerically: quadric decimation optimises visual fidelity, so it keeps slivers along
    creases. The decimated towel had a 639x edge-length ratio (0.206 mm to 131.8 mm), and
    explicit stability is set by the SMALLEST edge -- a 0.2 mm sliver in a 35 cm towel demands
    a timestep ~600x finer than the mesh's scale suggests. Every stiffness diverged.

    Physically, which matters more: a scan is one CRUMPLED configuration frozen in place. Cloth
    rest state is flat, and the rest state is what sets the strain in every triangle. Feeding a
    crumpled snapshot as the rest shape means the cloth starts with the folds built into its
    zero-energy configuration -- so it can never drape, because it is already "relaxed" in the
    shape it was scanned in.

    The scan supplies DIMENSIONS and appearance. The simulation mesh should be a regular grid,
    which is also what `build_flag_model` has always done for the synthetic flag.
    """
    ext = np.asarray(asset_mesh.extents, float) * float(scale)
    w, d = float(np.sort(ext)[-1]), float(np.sort(ext)[-2])   # two largest = the sheet
    return max(int(round(w / cell)), 4), max(int(round(d / cell)), 4), w, d


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


def build_cloth_model(nx, ny, cell, mass, tri_ke, tri_kd, edge_ke, edge_kd=0.1,
                      pos=(0, 0, 1.0), radius=0.01):
    """Flat rest state, uniform cells -- one edge length, so one stable timestep.

    `default_particle_radius` is NOT optional. Left unset, Newton's default is large enough
    that neighbouring particles of the same sheet overlap in the rest configuration, so the
    solver pushes them apart from the first substep and the cloth explodes -- measured 253.7 m
    of travel in 0.5 s of free fall, against the 1.228 m of actual free fall, with no ground
    and no contacts in the scene at all. `build_flag_model` sets it; that one line is the
    entire difference between a cloth that works and one that does not.
    """
    b = newton.ModelBuilder()
    b.default_particle_radius = float(radius)
    b.add_cloth_grid(pos=wp.vec3(*pos), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                     dim_x=int(nx), dim_y=int(ny), cell_x=float(cell), cell_y=float(cell),
                     mass=float(mass), tri_ke=float(tri_ke), tri_ka=float(tri_ke),
                     tri_kd=float(tri_kd), edge_ke=float(edge_ke),
                     edge_kd=float(edge_kd))
    b.color()
    return b.finalize()


def build_soft_model(V, T, density, k_mu, k_lambda, k_damp, pos=(0, 0, 1.0), radius=0.005):
    """Defaults copied from DiffSoft, which is a validated stable point for this solver.

    `radius` is the same trap the cloth hit, and the ratio to particle SPACING is what matters:
    DiffSoft uses radius 0.005 at cell 0.03, i.e. 1/6 of the spacing. A radius that is a large
    fraction of the spacing makes neighbouring particles overlap in the REST configuration, so
    the solver pushes them apart from the first substep and the body explodes.
    """
    b = newton.ModelBuilder()
    b.default_particle_radius = float(radius)
    mesh = newton.TetMesh(vertices=[wp.vec3(*v) for v in np.asarray(V, np.float32)],
                          tet_indices=np.asarray(T, np.int32).flatten().tolist())
    b.add_soft_mesh(pos=wp.vec3(*pos), rot=wp.quat_identity(), scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0), mesh=mesh, density=float(density),
                    k_mu=float(k_mu), k_lambda=float(k_lambda), k_damp=float(k_damp))
    b.color()
    return b.finalize()
