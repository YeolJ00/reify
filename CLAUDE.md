# CLAUDE.md

## What this project is

We make authored 3D assets **simulation-ready**: recover physical parameters **θ**
(density, stiffness, damping, friction) so an asset moves the way its material should.
Authored assets never physically existed — there is no footage and no ground truth — so a
video-prior VLM supplies the only available definition of "moves right," and the
simulator guarantees every output is physical.

**Core design: the video model is a judge, never a witness.** We never fit to generated
pixels. The simulator (Newton/Warp) produces candidate rollouts under different θ; the
judge (**Cosmos 3**) compares rendered candidates pairwise — "which moves more like
{material}?" — and we optimize θ against its preference plus a prior that pins absolute
scale. Hallucination cannot enter θ: the judge only ranks motions we manufactured.

## Non-negotiable design rules (each one bought with a failure)

1. **Pairwise only, never absolute scores.** Absolute yes/no scoring tracks motion
   magnitude, not material (J2 failure). Offsets and prompt framing cancel only in
   pairwise comparisons.
2. **Fixed target-material appearance in renders.** Same look across all candidates of a
   fit (only θ varies). Grey untextured sheets are out-of-distribution for the judge.
3. **Any judge change invalidates calibration.** Model, version, thinking mode, prompt,
   or render style changes ⇒ rerun the pairwise kill test (J2b) on the cached sweep
   before trusting any fit. Cheap (minutes); skipping it is how we get silently wrong.
4. **Watch the elite/final clips every run.** Known hacking ghost: "silk" = flappiest
   clip, "tarp" = stillest. If results sort by motion magnitude, stop and report.
5. **Run, don't assume.** Execute and show numbers/plots/clips before claiming success.
   Verify Cosmos 3 and Newton/Warp APIs against current docs; trained-in knowledge is stale.
6. **Deterministic, seeded, cached.** Renders and judge calls are the cost centers.

## State

- Render harness, judge harness: **done**.
- Absolute-score calibration (J2): **failed** → produced rules 1–2.
- Pairwise magnitude-matched calibration (J2b): **passed** → gate open.
- Multi-start CEM fit (J3): **running.** When it lands, review in this order: elite clips
  (rule 4) → did starts land in the known-correct parameter region per material? →
  per-parameter scatter across starts (agree = pinned, scatter = unconstrained) →
  objective traces. One markdown report.

## Current direction: the G-track (gradient MAP, no new models)

CEM validated the objective but won't scale with objects × parameters. Gradients scale.
The judge stays **frozen** — we backprop *through* it (autograd input-gradients, like
classifier guidance), never train it or any surrogate.

Objective: ascend  s(θ) + log p(θ)  where  s(θ) = pairwise logprob(A) − logprob(B),
A = differentiable-render(sim(θ)), B = a frozen reference clip (current best; promote
when decisively beaten — a ratchet). Averaged over both A/B orders. This keeps the
calibration properties of rule 1 inside a differentiable scalar.

Chain and its three links: ∂s/∂V through the frozen judge (**requires thinking disabled
for scoring** — a sampled trace has no gradient); ∂V/∂x needs a **differentiable
renderer** (nvdiffrast or PyTorch3D, same fixed appearance — new code, known tech);
∂x/∂θ via Warp autodiff (short windows, soft contact, clip/smooth through contact).
Reasoning-mode judging returns only as periodic *verification* of trajectory elites —
never as the driver.

Why hacking risk is low: θ is a ~5-d physical bottleneck; gradient ascent can only move
the clip along real cloth motions, not into adversarial pixel directions. Prior bounds +
rule-4 inspection + periodic reasoning-mode verification are the tripwires.

**G-milestones (cheapest kill first, strictly in order):**
- **G0** — rerun J2b in no-think pairwise mode (judge-mode change ⇒ rule 3). If it fails,
  gradient MAP through this judge is dead on calibration grounds; stop and report.
- **G1** — differentiable re-render of one cached rollout; check ∂s/∂pixels is finite/sane.
- **G2** — end-to-end ∂s/∂θ on the single flag, verified against finite differences;
  then one full gradient-MAP fit vs. the J3 CEM result on the same material, head-to-head.
- **G3** — only after G2 wins or ties: multi-object scene, per-object masks and
  factorized objectives; multi-start scatter as the uncertainty report throughout.

## Later (do not build ahead)

- Evidence seam: pluggable scorer registry; add cheap differentiable physics terms
  (e.g. static equilibrium). Generative-video evidence (i2v probes, continuations)
  returns here as extra terms if needed.
- Scene scale + active probing: scatter/uncertainty decides which probe to run next.

## Environment & layout

NVIDIA GPU; Newton/Warp (alpha, expect churn) + Cosmos 3; VRAM-tight ⇒ judge as a
separate pass over cached clips. Python, pinned requirements.

```
configs/
src/
  sim/        Newton wrapper: scene, rollout, θ
  render/     mp4 harness (done) + differentiable renderer (G1)
  judge/      Cosmos 3 pairwise comparator; no-think scoring mode (G0)
  optimize/   CEM (validated) + gradient-MAP ratchet loop (G2)
  evidence/   scorer registry [later]
  motion/     RETIRED tracker/spectral loss — reference only
scripts/      j2b_pairwise_test.py · j3_fit.py · g0..g2 per milestone
```
