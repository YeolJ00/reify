# CLAUDE.md

## What this project is

We make authored 3D assets **simulation-ready**: recover physical parameters **θ**
(mass, friction, restitution/damping, centre of mass, stiffness) so an asset moves the way
its material should. Authored assets never physically existed — there is no footage and no
ground truth — so a video-prior VLM (**Cosmos 3**) supplies the only available definition of
"moves right," and the simulator guarantees every output is physical.

**Core design: the video model is a judge, never a witness.** We never fit to generated
pixels. The simulator (Newton/Warp) produces candidate rollouts under different θ; the judge
compares rendered candidates and we optimise θ against its preference plus a prior.
Hallucination cannot enter θ: the judge only ranks motions we manufactured.

## Non-negotiable rules (each bought with a failure)

1. **A probe must be shown to depend on its parameter before it is trusted.** Sweep the
   parameter, report the correlation or threshold shift, *then* report a recovery. The
   `collide` probe was built to recover mass and measures nothing — a body shoved at fixed
   velocity slides `v²/(2μg)`, independent of mass. No amount of tuning could have fixed it.
2. **Check against invariants, not reference values.** Scale invariance, energy conservation,
   "does the reading respond to its input", two routes to one quantity agreeing. Every
   diagnosis made this way held; every one made by comparing against a handbook number had to
   be withdrawn. With no ground truth available, invariants are what we have.
3. **Detect divergence by a spatial bound, not `isfinite`.** An explicit solver reaches 10⁵ m
   long before `inf`. A cloth probe once reported ρ = −0.824 on a run that had exploded to a
   455 m footprint.
4. **Fixed target-material appearance in renders.** Magenta missing-texture surfaces are a
   different material than the one being asked about, not merely ugly.
5. **Any judge change invalidates calibration.** Model, version, thinking mode, prompt or
   render style ⇒ rerun the pairwise kill test before trusting a fit.
6. **Run, don't assume.** Print the trajectory before theorising about it. Five physically
   sensible hypotheses about a scale bug failed in a row; looking at the numbers found it.
7. **Deterministic, seeded, cached.** Renders and judge calls are the cost centres.

## State

### The simulator side is strong

- **Exact mesh contact** (`src/sim/mesh_contact.py`, `mesh_scene.py`). Ground contact tests
  mesh *vertices* against the plane — exact, since a linear function over a triangle is
  minimised at a vertex. Body–body uses `wp.mesh_query_point_sign_normal` in the *other*
  body's local frame, so each `wp.Mesh` is built once and never refitted. Replaced a sphere
  cover that was 2.28 mm from the true surface.
- **Per-body everything**: friction, mass, damping, contact stiffness. The **table is an
  object** with its own μ and cd, combined by geometric mean.
- Bounded tables, arbitrary initial orientation (`rot0`), exact reduced-coordinate revolute
  joints with per-body damping, real-asset tetrahedralisation.

### Probes

| probe | reads | result | script |
|---|---|---|---|
| tilt | friction (vs table) | 0.40° residual, 3 cm–116 cm | `bigscene_sim.py` |
| balance | **mass** of a real asset | 7/7, brackets 0.182 kg at [0.18, 0.22] | `probe_balance.py` |
| bounce | damping | ρ = −0.995 | `probe_bounce_settle.py` |
| settle | damping | ρ = +0.973 | `probe_bounce_settle.py` |
| spin | friction (vs table) | ρ = −0.952 | `probe_spin_stack.py` |
| stack | friction **between objects** | ρ = +0.994, residual 2.41° | `probe_spin_stack.py` |
| cloth drape | stiffness | ρ = −0.786, saturated at soft end | `probe_deformable.py` |
| soft press | stiffness | **diverges** | `probe_deformable.py` |

`collide` (×3) and `drop` are **retired** — see `docs/PROBE_SPIN_STACK.md`. Collide's
observable is provably mass-independent; drop is superseded by bounce. Both are absent from
`PROBE_MASK`, not zero-weighted, because a retired probe still costs a sim + render + judge
call per iteration. `shove` remains but predates the mesh solver and is unverified.

**`PROBE_W` is provisional**: the two retired probes held the only measured Fisher weights, so
every surviving probe now starts at the unmeasured mean until the yes-region sample is re-run.

### The judge is the bottleneck, and is unchanged

Approves 52.8% of clips at p > 0.90. Correlates −0.02 with density, −0.07 with damping,
−0.25 with friction across a 10× sweep. Names materials 4/4 correctly but answers bounce
counts 1–2 against a true 1–21. Three fits of the same object gave densities 244 / 188 / 356.

**We can now manufacture excellent evidence and still cannot read it off a video.**

## Findings that changed the plan

- **Magnitude observables are not inherently confounded.** The rule that only thresholds work
  came from measurements taken *across objects*, where shape dominates (a drop probe gave
  ρ = −0.186 over 14 objects). Holding object and excitation fixed and sweeping only the
  parameter, the same observable gives −0.995. Magnitudes are fine *within* an object — which
  is the regime per-object θ already puts us in.
- **Solver parameter sets are validated jointly, not individually.** `configs/flag.yaml` is a
  stable point; changing mass alone (0.05 → 0.02) diverges. Stability couples
  `dt·sqrt(ke/m) < 0.3` and `kd/m·dt << 1`.
- **`default_particle_radius` must be set** for cloth and soft bodies. Unset, neighbouring
  particles overlap in the *rest* configuration and the body explodes — 253.7 m of travel in
  free fall with no contacts, against a correct 1.228 m.
- **A scan is a crumpled snapshot, not a rest state.** Cloth rest shape sets the strain in
  every triangle, so simulating a scan's own triangles bakes the folds into the zero-energy
  configuration and it can never drape. The scan gives dimensions and appearance; the sim
  mesh should be a regular grid.
- **Coarsening a timestep to speed a render changes results.** Balance went 7/7 → 4/7 when
  substeps were cut 60 → 24 for a render pass.
- **Inertia carried mass twice.** `set_masses` stored `G = G_unit · m` while `integrate_6dof`
  forms `Iw = R (m·G) Rᵀ`, so inertia was `m²·G_unit` — for a 0.1 kg body, ten times too
  little, and ten times the angular acceleration. Fixed; it also improved the tilt probe
  (residual 0.79° → 0.40°).
- **Relative motion must be measured in the moving body's own frame.** The stack probe took
  `top − base` in world coordinates, so the base rotating with the ramp read as sliding and
  every case collapsed the instant the table moved (ρ +0.933 → +0.994).
- **Contact coefficients combine as a geometric mean**, so a body slips at
  `atan(sqrt(μ_body·μ_table))`, not `atan(μ_body)`.
- **A body pinned at its own centre of mass is in neutral equilibrium.** `MeshProbeScene` puts
  the origin at the vertex mean, so `set_hinge` without `pivot_local` gives a balance that
  slams to its stop on any imbalance and reports no magnitude. The pivot rise above the CoM
  *is* the instrument's sensitivity.
- **A bowl is a shell.** Near its axis the lowest vertices are the outside of its base; a
  payload seated there starts inside the floor and gets ejected on contact.

## Next, in order

1. **Ask the judge a threshold question.** Several probes are now one-bit — *which pan went
   down*, *did it slide at this angle*. That is a question a VLM plausibly answers, unlike
   bounce counting, and the balance gives ground truth to score it against. Cheapest decisive
   test available; do this before building anything else.
2. **Fix the soft-body probe.** Copy `DiffSoft`'s validated set (`k_mu=6000, lam_ratio=2,
   k_damp=10, density=800, substeps=64, radius = spacing/6`) verbatim, then sweep only `k_mu`.
   Tetrahedralisation is already verified.
3. **Finer cloth grid.** ρ = −0.786 saturates because a 7×5 grid cannot represent a fold.
   Needs a re-derived stable (mass, cell, ke, kd) point, not a changed cell alone.
4. **Render bounce and settle.** Each needs a `*_render_prep.py` exporting poses in
   `blender_render_scene.py`'s format — `balance_render_prep.py` shows the pattern, including
   the COM-offset convention.
5. **Per-object focus in the fit**: full-frame global term alongside the per-object crops.
   `multiprobe_build.py` already writes `_FULL.mp4`; `joint_fit.py` ignores it.

## Environment & layout

NVIDIA GPU (use one, don't hold idle ones); Newton 1.4.0 / Warp 1.15.0 (alpha, expect churn)
+ Cosmos 3; VRAM-tight ⇒ judge as a separate pass over cached clips.
Python at `/home/jooyeolyun/anaconda3/envs/warp/bin/python`.
Blender at `/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender`.
`SolverVBD` gradients are exactly 0.0 in newton 1.4.0 — use `SolverSemiImplicit` for anything
differentiable.

```
configs/
src/
  sim/        mesh_contact · mesh_scene · deformable · probe_scene (legacy sphere cover)
  render/     Cycles harness (RENDER_SCALE, CAM_PULL, TABLE_SX/SY) · crop · camera · views
  judge/      Cosmos 3 pairwise + absolute; fp32 head, full-frame decode
  optimize/   CEM · SPSA with per-object θ
scripts/      probe_*.py · bigscene_sim.py · joint_fit.py · iter_report.py
docs/         PROBE_*.md · SCALE_RANGE.md · MESH_CONTACT.md · BIGSCENE.md · iters/
```

`scripts/iter_report.py [RUN] --max-clips N` emits one page per object: θ per iteration, the
per-probe plus/minus scores, and the clip pairs that produced them.
