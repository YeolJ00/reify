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

## The bug: `default_particle_radius`

`build_flag_model` sets `builder.default_particle_radius = 0.01`. I did not. That one line is
the whole difference. Left unset, Newton's default is large enough that neighbouring particles
of the same flat sheet overlap in the REST configuration, so the solver pushes them apart from
the first substep.

Bisected in free fall with no ground and no contacts in the scene, where a rest-state cloth
should translate exactly 1.23 m in 0.5 s:

| configuration | max travel |
|---|---|
| FlagSim values, radius 0.01 | **1.228 m** — correct |
| same, radius unset | **253.7 m** |
| same, my edge_kd 1e-4 | 1.228 m — harmless |

Two earlier diagnoses were wrong and are recorded as such: the decimated scan's 639× edge-length
ratio (a clean uniform grid diverged identically) and the claim that `pymeshlab` was needed for
isotropic remeshing (no new dependency is involved).

## Second finding: flag.yaml is validated as a SET

With the radius fixed, deviating from `configs/flag.yaml` on mass (0.05 → 0.02) and cell
(0.05 → 0.03) still diverged at substep 22 with no ground present. The config's own comment
gives the reason — stability couples the three: `dt·sqrt(ke/m) < 0.3` and `kd/m·dt << 1`.
Changing one at a time is not safe; the point is validated jointly.

## Cloth: runs, reading is weak

Towel 36×26 cm as a 7×5 uniform grid, `flag.yaml` verbatim apart from the swept `tri_ke`:

| tri_ke | spread | height |
|---|---|---|
| 600 | 19.04 cm | 0.00 cm |
| 1500 | 19.04 cm | 0.00 cm |
| 5000 | 19.04 cm | 0.00 cm |
| 15000 | 16.08 cm | 6.81 cm |
| 40000 | DIVERGED | |

**ρ(log tri_ke, spread) = −0.786**, and it converges — but the three softest values are
identical, so the probe only discriminates at the stiff end. The likely cause is resolution: at
cell 0.05 the towel is 7×5 cells, which cannot represent a fold. Going finer requires re-deriving
a stable (mass, cell, ke, kd) point rather than changing cell alone.

## Soft body: still diverges, 5/5

Unresolved. The tetrahedralisation itself is verified (660 verts, 2070 tets, all positive
volume), and the particle radius is now set, so the remaining suspect is the same one the cloth
had — the tet material parameters need a validated stable point analogous to `flag.yaml`.
`DiffSoft` (`src/sim/diff_soft.py`) has one: `k_mu=6000, lam_ratio=2.0, k_damp=10.0,
density=800, substeps=64`, on a procedural cube. That is the set to copy verbatim next, exactly
as `flag.yaml` was for cloth.

## Status

Nothing in the cloth row should be quoted as a stiffness measurement yet; it discriminates only
between "limp" and "stiff". Soft body produces no reading at all.
