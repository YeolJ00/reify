# simulation-assetization

Per-instance inverse simulation from video priors: recover physics parameters θ for
roughly-placed simulation assets so a differentiable Newton simulation reproduces the
physically-realizable part of an observed motion. See `CLAUDE.md` for the full plan.

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
