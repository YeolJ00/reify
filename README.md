# simulation-assetization

Per-instance inverse simulation from video priors: recover physics parameters θ for
roughly-placed simulation assets so a differentiable Newton simulation reproduces the
physically-realizable part of an observed motion. See `CLAUDE.md` for the full plan.

**Docs:** `docs/report.html` — self-contained milestone report (open in a browser);
`docs/PROJECT_LOG.md` — technical log for the working agent. Regenerate the HTML with
`python docs/build_report.py` after new results.

## Status — M0/M1 complete (2026-07-21)

- **M0**: cloth flag (25×17 particles) pinned on the hoist edge, constant-wind
  normal-pressure force on faces, 60 frames × 32 substeps on GPU.
  `python scripts/run_forward.py [--solver semi_implicit|vbd]`
- **M1**: `d(loss)/d(wind_strength)` through the full rollout via `wp.Tape`,
  verified against central finite differences.
  `python scripts/check_grad.py [--solver semi_implicit|vbd]`
  - `semi_implicit`: **gradient verified** (rel. err 3e-3 vs FD at h=0.3).
  - `vbd`: tape gradient is exactly **zero** (newton 1.4.0) → not differentiable;
    FD is clean, so the CEM fallback is used for VBD.
- **M2 (single param)**: recover wind strength from a synthetic target
  (theta_true=15, theta0=8).
  `python scripts/recover_synthetic.py --method grad` (Adam on tape gradient) or
  `--method cem` (zeroth-order fallback, works with VBD).
  - grad, 80 iters: recovered **15.088** (0.59% err), 104 s.
  - cem, 15 iters x pop 16: recovered **14.992** (0.05% err), 90 s.

## M3 — full theta + identifiability (2026-07-21)

theta = [wind a0, a1, a2 (Fourier forcing), gravity_z, log tri_ke, log tri_kd,
log edge_ke, log mass], all differentiable through the rollout
(`src/sim/theta_sim.py`: theta is written into the model arrays by warp kernels
inside the tape; newton 1.4.0 solvers read gravity/materials from arrays at
every launch, so adjoints flow).

- `scripts/recover_full.py [--method grad|cem] [--init-from <npz>]`
- `scripts/probe_identifiability.py` — FD Hessian + 2D loss slice at theta_true.

Findings:
- **The adjoint is correct but the landscape is rugged far from the target**:
  FD-vs-tape matches to a few % near theta_true (and for dominant components
  anywhere), but far away FD does not converge as h shrinks — chaotic flapping
  makes the local gradient noisy. Plain Adam from a far init stalls
  (loss plateaus ~3e-2); CEM gets into the basin (~1.5e-3); the intended
  pipeline is **CEM (global) -> Adam (polish)**.
- **Identifiability (Gauss-Newton J^T J at theta_true, PSD)**: condition number
  3.3e5. Strong: log_mass, log_tri_kd (RMS sensitivity ~0.4, their ratio is the
  stiffest direction), then log_tri_ke, gravity_z. Sloppy (50-250x weaker): all
  three wind-forcing coefficients and log_edge_ke (bending ~unobservable).
  Forcing does not exactly alias with material, but lives in the near-null
  subspace -> wind wants a prior/posterior treatment, materials support a point
  estimate. (The naive FD loss-Hessian shows spurious negative eigenvalues —
  the basin is narrower than the FD step in stiff directions; use the GN form.)
- **Stability cliff in theta-space**: reducing mass ~10% below true (at fixed
  dt) explodes the semi-implicit sim — the loss landscape contains a numerical
  cliff, visible in the 2D slice. CEM tolerates it (NaN sorts last).
- **Levenberg-Marquardt (FD trajectory Jacobian, `src/optimize/lm.py`) is the
  workhorse**: it handles the 3e5 conditioning that defeats Adam (uniform steps
  overshoot stiff directions / undershoot sloppy ones). One run from the far
  init reached loss 3.3e-8, recovering the target trajectory exactly — but
  landing on the scaling orbit: mass, tri_ke, tri_kd, wind all off by the SAME
  factor 1.13, gravity exact (0.02%). Direct empirical proof of the gauge
  freedom. Needs multi-start: LM is a basin lottery (GPU atomic float
  nondeterminism alone routes identical runs to different minima; a rerun
  stalled at 5.5e-4, and LM seeded from the CEM point hit a worse basin).
- **Gauge fixing works**: `recover_full.py --method lm --fix log_mass` (density
  known, e.g. from an asset prior) recovers ALL observable params to <4%:
  wind 0.08/0.99/3.8%, gravity 0.33%, tri_ke 0.22%, tri_kd 0.15%. Only edge_ke
  collapses (to 0) — consistent with its ~250x-below-leading sensitivity:
  bending is unobservable in full-trajectory MSE for this scene.
- **M3 decision (point estimate vs posterior)**: the method needs either a
  scale anchor (known density/mass -> sharp point estimate) or a posterior over
  the scaling direction + unobservable bending. Forcing does not alias with
  material *shape* parameters, but shares the overall scale gauge with them.

## Setup

```bash
conda create -n warp python=3.11 -y
conda run -n warp pip install -r requirements.txt
```

Requires an NVIDIA GPU. Newton API note: the PyPI package is `newton`
(`newton-physics` is a renamed placeholder). `SolverVBD` requires
`ModelBuilder.color()` before `finalize()`.

## Findings log

- `add_cloth_grid` vertex indexing is `y * (dim_x + 1) + x`; `fix_left` pins the
  `x == 0` column by zeroing particle mass.
- Semi-implicit stability: explicit damping needs `tri_kd * dt / mass << 1`
  (tri_kd=100 at mass=0.1, dt=1e-3 exploded; config now keeps the ratio ≈ 0.1).
- Wind is not built into newton solvers; we apply it as an external per-triangle
  force into `state.particle_f` before each substep (differentiable warp kernel,
  `src/sim/rollout.py::wind_force_kernel`).
- VBD is ~35× slower per substep than semi-implicit here, but implicit (could take
  larger steps). Diffsim examples in newton itself all use `SolverSemiImplicit`.
- Plain CEM collapsed sigma by iter 2 and stalled 8% off; decaying additive noise
  in `cem_1d` (default `sigma0/4`, annealed to 0) fixed it (0.05% final error).
