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

## Two wrong diagnoses, recorded because they were confidently stated

**"Mesh quality — 639x edge-length ratio."** True of the decimated scan, and it did motivate a
real improvement (below), but it was not the cause: a clean 25x19 uniform grid with a single
edge length diverges identically.

**"pymeshlab is needed for isotropic remeshing."** Reaching for a dependency instead of
questioning the premise. No new dependency is needed, and the premise was wrong twice over --
see below.

## The improvement that survives both

`cloth_grid_from_asset` sizes a **uniform grid** from the scan's bounding box rather than
simulating the scan's own triangles. Beyond the numerics, this is the physically correct
choice: a scan is one **crumpled configuration frozen in place**, and a cloth's rest state is
what sets the strain in every triangle. Feeding a crumpled snapshot as the rest shape bakes the
folds into the zero-energy configuration, so the cloth can never drape -- it is already
"relaxed" in the shape it was scanned in. The scan supplies dimensions and appearance; the
simulation mesh should be regular, which is what `build_flag_model` has always done.

## Where the fault actually is

Isolation test, 25x19 grid, 64 substeps, 0.5 s:

| configuration | stable | max displacement |
|---|---|---|
| no ground, `contacts=None` | no | **79.3 m** |
| ground, `contacts=None` | no | 74.9 m |
| ground, `model.collide()` | no | 90.5 m |

A flat cloth at its rest configuration has zero internal strain and, in free fall with no
contacts, should simply translate 1.2 m in 0.5 s. It moves 79. So the fault is neither contact
handling nor mesh quality but the **model construction itself** -- something in the
`add_cloth_grid` parameters or the step loop is injecting energy from the first substep.

## Next step, concrete

`FlagSim` (`src/sim/rollout.py`) + `build_flag_model` (`src/sim/scene.py`) run this same solver
on this same kind of grid and are known-good, with a working config at `configs/flag.yaml`.
Diff the builder call and the step loop against them -- particularly the per-particle `mass`
(0.004 kg here), `edge_ke`/`edge_kd` scaling, whether `b.color()` belongs before `finalize()`
for `SolverSemiImplicit`, and the preallocated-state loop versus the two-state ping-pong used
here. That is a direct comparison against working code, not an open-ended debug.
