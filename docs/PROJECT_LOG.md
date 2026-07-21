# Project log (Claude-facing)

Technical log of everything done, found, and decided. Human-facing narrative:
`docs/report.html` (self-contained, keep both in sync). Date: 2026-07-21.

## Environment

- conda env `warp` → run everything with `/home/jooyeolyun/anaconda3/envs/warp/bin/python`,
  prefix `CUDA_VISIBLE_DEVICES=<free gpu>` (shared 8× RTX A6000 box; check `nvidia-smi`).
- `newton==1.4.0`, `warp-lang==1.15.0`, `opencv-python-headless==5.0.0.93`; pinned in
  `requirements.txt`. PyPI package is `newton` — `newton-physics` is a renamed placeholder.

## Newton 1.4.0 API facts (verified against installed source, not memory)

- `ModelBuilder.add_cloth_grid(...)`: vertex index = `y * (dim_x + 1) + x`;
  `fix_left` pins the `x == 0` column by setting particle mass 0 and clearing
  `ParticleFlags.ACTIVE`.
- `SolverVBD` requires `builder.color()` before `finalize()`.
- `finalize(requires_grad=True)` makes all model arrays differentiable, including
  `model.gravity` which is `wp.array(vec3, shape (1,))` and is read *inside* the
  particle-integration kernel every launch (`_src/solvers/solver.py:88`:
  `v1 = v0 + (f0*inv_mass + gravity*nonzero(inv_mass))*dt`) → gravity is an
  acceleration; internal/wind forces scale by 1/m. Basis of the scale gauge freedom.
- `tri_materials` layout: `(ke, ka, kd, drag, lift)` per triangle;
  `edge_bending_properties`: `(ke, kd)` per edge.
- No built-in wind. We add per-triangle aerodynamic normal pressure
  `f = drag * area * dot(n̂, v_wind − v_tri) * n̂` into `state.particle_f` before each
  substep (`src/sim/rollout.py::wind_force_kernel`, theta version in `theta_sim.py`).
- Determinism: `wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN` must be set
  before solver construction (newton reads it in solver `__init__` and marks its kernel
  modules); our own atomic-emitting modules need explicit
  `wp.set_module_options({"deterministic": ..., "deterministic_max_records": 0}, module=...)`.
  Verified bitwise-identical trajectories across processes. **Cost: 7×** (0.40 s → 2.81 s
  per 1920-substep rollout) → config default `sim.deterministic: false`; enable for
  verification runs only.

## Numerical stability (semi-implicit)

- Explicit damping limit: need `tri_kd * dt / mass ≪ 1`. tri_kd=100 @ mass 0.1, dt 1e-3
  exploded (300 m excursions, finite but garbage). Config keeps ratio ≈ 0.1
  (mass 0.05, tri_kd 10, substeps 32 → dt 5.2e-4).
- Stability cliff in theta space: mass ~10 % below true (at fixed dt) explodes the sim.
  The loss landscape contains this cliff right next to the optimum (visible in
  `outputs/identifiability.png` right panel). CEM tolerates it (NaN sorts last);
  LM rejects non-finite steps.

## Gradient findings

- M1: `d(loss)/d(wind)` through the full rollout via `wp.Tape` matches central FD to
  0.3 % (semi-implicit). **VBD tape gradient is exactly 0.0** in newton 1.4.0 — not
  differentiable; use zeroth-order (CEM/LM-FD) with VBD.
- The 8-dim theta adjoint is correct (dominant components verify to 2–9 % near
  theta_true) but **far from the target the landscape is rugged/chaotic**: FD does not
  converge as h shrinks; the exact local gradient is real but noisy. Plain Adam from a
  far init stalls (~3e-2 plateau); Adam also fails from a CEM-refined init because
  uniform step sizes overshoot stiff directions (condition number 3.3e5).

## theta parameterization (M3)

`theta = [wind_a0, wind_a1, wind_a2, gravity_z, log_tri_ke, log_tri_kd, log_edge_ke,
log_mass]` (`src/sim/theta_sim.py`). Wind speed s(t) = a0 + a1 sin(2πft) + a2 cos(2πft),
f = 1 Hz fixed, direction fixed. Positive params in log space. Fill-kernels write theta
into model arrays *inside the tape* each rollout. YAML gotchas: `true:` key parses as
bool (renamed to `true_values:`); `5.0e3` without sign parses as string (float() casts).

## Identifiability (M3 probe)

Gauss-Newton J^T J from FD trajectory Jacobian at theta_true (PSD; the naive FD
loss-Hessian shows spurious negative eigenvalues because the basin is narrower than the
FD step in stiff directions — don't use it):

- RMS trajectory sensitivity per unit param: log_mass 0.45, log_tri_kd 0.40,
  log_tri_ke 0.083, gravity_z 0.067, wind_a0 0.010, wind_a2 0.0058, wind_a1 0.0054,
  log_edge_ke 0.0017. Condition number 3.3e5.
- Stiffest direction pairs kd against mass (their ratio); wind forcing lives in the
  sloppy subspace; **edge bending is effectively unobservable** in full-trajectory MSE.
- **Scale gauge**: free LM converged to loss 3.3e-8 with mass, tri_ke, tri_kd, wind all
  scaled by the SAME 1.13 factor and gravity exact (0.02 %) — empirical proof that only
  force/mass ratios are observable; gravity (acceleration) is the only absolute anchor
  and it cannot fix the scale.
- **Decision**: gauge-fixed point estimate (pin density/mass from an asset prior) or a
  posterior over scale + bending. `recover_full.py --fix log_mass` recovers all
  observable params to <4 % (wind 0.08/0.99/3.8 %, gravity 0.33 %, tri_ke 0.22 %,
  tri_kd 0.15 %); edge_ke collapses to 0 (unobservable, as predicted).

## Optimizers

- `src/optimize/cem.py`: 1-D and diagonal n-D CEM with decaying additive noise
  (plain CEM collapses sigma by iter ~2 and stalls; noise0 = sigma0/4 annealed to 0
  fixed it). Returns best-ever sample, not final mean (mean drifts along sloppy valley).
- `src/optimize/lm.py`: LM on FD trajectory Jacobian (16 rollouts/iter for 8 params);
  solver-agnostic (works with VBD). Handles the 3e5 conditioning. **Basin lottery**:
  identical runs can land in different minima (GPU atomic nondeterminism alone flips
  basins: 3.3e-8 vs 5.5e-4 across reruns) → `recover_full.py --starts N` multi-start;
  best-of wins. CEM→LM chaining was WORSE than LM from the neutral init (CEM's drift
  in sloppy directions, e.g. edge_ke → 52, poisons the basin).
- Multi-start LM (5 starts, --fix log_mass, 423 s): 3/5 starts reach the global basin
  (final loss ~5e-10); 2 stall in local minima — including the *neutral* config init,
  while perturbed starts won (basin lottery confirmed; multi-start is the fix, and
  ranking by final loss is a reliable selector). Winner recovers **all params to
  ≤0.07 %**, even edge_ke to 1.31 % — at loss 5e-10 the bending signal, ~250× below
  leading sensitivities, finally counts.

## M4 stage 1 (tracked motion from a rendered rollout)

Pipeline (`scripts/run_m4_pipeline.py`): matplotlib rasterizer (pixel-exact, painter
depth sort, static random per-triangle grayscale texture) → chained pyramidal LK with
forward-backward gate (fb < 1 px; = step-8 rejection at tracker level) → tracks attached
to cloth via frame-0 barycentric coords (frontmost triangle wins) → LM on 2D pixel
residuals through `project_bary_points_kernel` (differentiable pinhole, FD-verified to
0.6 %) → gauge-fixed log_mass, multi-start.

- Camera at ~30° azimuth so in-plane and cross-flow motion both project.
- Result (3 starts, 333 s): all starts converge to the same plateau
  ~6e-5 ≈ **4 px RMS — the LK tracking noise floor**, not an optimizer failure.
  Recovery vs true: wind_a0 13.8 %, gravity 19 %, tri_ke 25 %, tri_kd 30 %;
  wind harmonics poor, edge_ke → 0. With exact 3D supervision the same optimizer
  reaches ≤0.07 % — so the accuracy limit has moved from optimization to
  *observation quality*. Levers: better tracker (CoTracker), sub-pixel refinement,
  multi-view, longer horizon, robust/weighted 2D loss.
- Visual check: recovered-sim reprojections lie on top of LK tracks across the whole
  deformed flag (`outputs/m4_stage1.png`).

## Script map

- `scripts/run_forward.py` — M0 rollout (`--solver semi_implicit|vbd`)
- `scripts/check_grad.py` — M1 FD-vs-tape check (exit 1 + CEM advice on failure)
- `scripts/recover_synthetic.py` — M2 single-param (`--method grad|cem`)
- `scripts/recover_full.py` — M3 8-dim (`--method grad|cem|lm --fix ... --starts N --init-from ...`)
- `scripts/probe_identifiability.py` — FD Hessian + 2D slice (use GN variant in
  `outputs/identifiability_gn.npz` for conclusions)
- `scripts/run_m4_pipeline.py` — M4 stage 1 end-to-end

## Open items / next

- M4 stage 2: real/generated video — needs an i2v model choice (step 3) + initial-frame
  render of the real asset; tracker upgrade to CoTracker if LK is insufficient on real
  footage.
- Occlusion in the renderer is painter-sorted only; folds may confuse LK on harsher
  motions — the fb gate currently absorbs this by killing tracks.
- eval/ metrics harness still a stub; formalize recovery-error metrics across seeds.
