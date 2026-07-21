# CLAUDE.md

## Project

Per-instance **inverse simulation from video priors**. Given a roughly-initialized
simulation asset (geometry and placement fixed, physics unknown) and a motion
observation, recover simulation parameters **θ** that reproduce the *physically
realizable* motion. The simulator is a **hard constraint**: the output can only be a
valid Newton simulation, so unphysical components of the observation are discarded by
construction rather than fitted. The goal is not to match the observation but to
recover the closest physical motion to it (projection onto the physical manifold).

We are currently solving each scene by **per-instance optimization** of θ through
Newton — no trained model. An amortized generator is an optional later escalation, not
the foundation. 

### What θ is (the optimization variables)
- **Per object:** constitutive type (`rigid` / `cloth` / `soft` / `articulated`) plus its
  continuous parameters — density, stiffness, damping, friction.
- **Per object:** initial linear and angular velocity.
- **Global:** gravity, and a **low-dimensional forcing field** (a handful of spatiotemporal
  Fourier modes, or a small field `f(x, t)` — never a dense per-cell wind volume).
- **Placement is INPUT, not optimized.** Fixing the discrete / contact-coupled degrees of
  freedom keeps θ in the smooth, gradient-friendly regime.

## Full system architecture

The complete inference pipeline. Each step is tagged with the milestone it enters; understand
the whole thing, but only build what the current focus calls for.

```
1. real 3D asset  → assemble rough multi-object scene S      [scale — after single-object M4]
2. render initial frame I0 from S                            [M4]  (forward render, not differentiable)
3. i2v(I0) → V*   (motion prior)                             [M4]  (image-to-video model)
4. extract motion from V*: point tracks / optical flow      [M4]  (e.g. CoTracker / flow)
5. propose θ      (per-instance optimization)               [now] (optional generator: later)
6. Newton(S, θ)   → 3D trajectory                           [now] (the differentiable sim)
7. project sim points to 2D via differentiable camera       [M4]  (differentiable projection)
8. match projected tracks to V* tracks; reject high-residual [now, synthetic] / [M4, video]
9. update θ (gradient + CEM); repeat                        [now]
   → output: θ and its simulation (editable physical animation)
```

Mapping to what the pipeline needs: **real asset data** = step 1, **multi-object** = steps 1 & 6,
**image-to-video** = step 3, **tracking / optical flow** = step 4, **differentiable rendering /
projection** = step 7. All are core; they are sequenced after the inverse-signal core is proven.

## Current focus — M0 / M1 (build only this)

Prove the simulator gives a usable inverse signal on a target we control. Run steps 6, 8, 9
against a **synthetic** target; steps 1–4 and 7 are deferred (do not build them yet).

1. **M0 — one rollout.** Install Newton + Warp, build one cloth flag pinned on one edge with a
   constant wind force on its faces, simulate T steps, save the vertex trajectory. Confirm it
   runs on GPU and states are readable.
2. **M1 — gradient sanity check.** Put a scalar loss on the trajectory (distance to a target
   rollout generated with a different wind strength) and compute `d(loss)/d(wind_strength)`
   through the rollout. **Verify it against finite differences.** If the gradient is wrong or
   unusable through the chosen solver, report it and set up a zeroth-order (CEM) fallback.

Definition of done: a runnable script recovering a single known parameter from a synthetic
target, with a printed loss curve and recovered value, plus a finite-difference check on the
gradient (or a working CEM loop if gradients are unusable).

Everything tagged `[M4]` or `[scale]` in the architecture above is deferred. Understand it, but
if a task seems to require building one of those modules now, **stop and flag it** instead.

## Build order (roadmap)

- **M0** one differentiable rollout on GPU.  (steps 6)
- **M1** validated gradient, or CEM fallback, w.r.t. one parameter.  (steps 6, 8, 9)
- **M2** recover a single parameter from a synthetic target.  (steps 6, 8, 9)
- **M3** recover the full θ vector; probe identifiability (do forcing and material trade off?).
  This result decides whether the method is written as a point estimate or a posterior.
- **M4** replace the synthetic target with tracked motion — first from a rendered sim we control
  (steps 2, 4, 6, 7, 8), then from a real / generated video (step 3 added).
- **Scale** multi-object scenes and real asset data.  (step 1)  — only after single-object M4 holds.
- **Optional** amortized generator + self-supervised degrade-restore prior over θ.  (replaces
  step 5's optimizer with a learned model) — only if forward-pass inference is wanted.

## Working rules

- **Run, don't assume.** Execute every sim and script and show the real output (numbers, plots)
  before claiming anything works. Never report success on unexecuted code.
- **Newton is alpha; do not trust remembered APIs.** Verify install steps and API signatures
  against the currently installed package / current docs. Assume training-data knowledge of
  Newton's API may be stale or wrong.
- **Verify every gradient with finite differences** before trusting it — especially through contact.
- **Synthetic-first.** Targets are generated by our own simulator with a known θ_true, so ground
  truth is exact. Log θ_true and recovered θ side by side. No video until M4.
- **Small, verifiable increments.** Each change ends in a runnable script that prints a number or
  writes a plot. One working step beats a large speculative framework.
- **Config-driven.** Scene and sim parameters live in `configs/`, not hardcoded.
- **Reproducible.** Seed all RNG; make rollouts deterministic where the solver allows.

## Environment

- Requires an **NVIDIA GPU**.
- **Newton** (physics engine on **Warp**, differentiable, currently alpha/beta). Expect API churn.
- Use the **VBD** solver for cloth. Confirm early whether gradients flow through it; if not, CEM.
- Python. Pin versions in `requirements.txt` once the environment works.
- Name the conda env as "warp". 
- The NVIDIA warp repository is under active development.

## Repo layout

Dirs for the full system; those past the current focus are stubs for now.

```
configs/          scene + sim parameters (yaml)
src/
  data/           asset loading + scene sampling (real assets)      [scale]
  sim/            Newton wrapper: build scene, rollout, expose θ    [now]
  motion/         motion loss; synthetic now, video tracks later    [now / M4]
  video/          i2v model + initial-frame render                  [M4 — stub]
  track/          point tracking / optical flow on video            [M4 — stub]
  render/         camera + differentiable projection                [M4 — stub]
  optimize/       per-instance loop (gradient + CEM)                [now]
  generate/       optional amortized generator + self-sup prior     [optional — stub]
  eval/           metrics harness                                   [M2+]
scripts/
  run_forward.py        M0: forward rollout, dump trajectory
  check_grad.py         M1: gradient vs finite-difference check
  recover_synthetic.py  M2/M3: recover θ from a self-generated target
tests/
```
