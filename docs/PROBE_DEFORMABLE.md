# Cloth and soft body — infrastructure works, probes diverge

## What works and is verified

**`src/sim/deformable.py`** — the first path in this repo from a *scanned asset* to a
deformable model. Everything before it was procedural: `FlagSim` builds a cloth grid,
`DiffSoft` an `add_soft_grid` cube. Newton's `add_cloth_mesh` and `add_soft_mesh` accept real
geometry and were unused.

- `cloth_from_mesh` — decimates the towel scan from **101142 → 2803 faces**. Verified.
- `tets_from_mesh` — voxelises a closed mesh and splits each interior voxel into 5 tets, with
  the split parity alternating by voxel index sum so neighbours agree on their shared diagonal
  (without it the mesh has cracks). Rubber duck → **660 verts, 2070 tets, all positive volume**.
  Verified.
  - Interior test uses **Warp's mesh SDF**, not `trimesh.contains` (needs `rtree`, absent) and
    not `voxelized().fill()` (needs `scipy`, absent). Containment also gives a genuinely solid
    body — surface voxels alone tetrahedralise a shell, and a shell has no bulk modulus.
- A **divergence guard** on a spatial bound rather than `isfinite`. An explicit solver past its
  stability limit reaches 10^5 m long before it reaches `inf`, so an `isfinite` check reports
  success on a blown-up run. The first version of this probe did exactly that and reported a
  towel with a 455 m footprint and ρ = −0.824, which looked like a working reading.

## What does not work

Both probes diverge at every stiffness tested, at 256 substeps (dt = 6.5e-5 s):

| probe | sweep | result |
|---|---|---|
| cloth drape | tri_ke 20 → 1620 | DIVERGED, 5/5 |
| soft press–release | k_mu 1e3 → 8.1e4 | DIVERGED, 5/5 |

**Cause: mesh quality, not solver settings.** The decimated towel has an edge-length ratio of
**639×** (0.206 mm to 131.8 mm) and a worst aspect ratio of 17. Explicit integration is stable
only below a timestep set by the *smallest* edge, so a 0.2 mm sliver in a 35 cm towel demands
a step ~600× finer than the mesh's overall scale suggests. Quadric decimation preserves flat
regions as large triangles and keeps slivers along creases — it optimises for visual fidelity,
which is the opposite of what a solver needs.

## What would fix it

**Uniform remeshing, not decimation.** The cloth needs roughly equilateral triangles at one
target edge length. `trimesh` has no isotropic remesher; options are `pymeshlab`
(`meshing_isotropic_explicit_remeshing`), `open3d`, or subdividing a fitted grid and projecting
onto the scan. Any of these is a dependency decision rather than a code problem.

The same applies to the soft body: the voxel tetrahedralisation is uniform *by construction*
(all tets come from a regular grid), so the duck should be better behaved than the towel — its
divergence is more likely the drop height and contact stiffness against `add_ground_plane`,
which is a separate tuning pass that was not reached.

## Honest status

The asset→deformable path is built and its two hard parts are verified. The probes do not yet
produce a reading. Nothing here should be quoted as a stiffness measurement.
