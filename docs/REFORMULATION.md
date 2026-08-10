# Reformulation — what this is, what we measured, what to do next

## What we are doing

Authored 3D assets have geometry but no physics. They never existed, so there is no footage
and no ground truth. We give them physical parameters **θ** by:

1. the **simulator** generating candidate motions under different θ (every candidate is
   physically valid by construction — the simulator is a hard constraint),
2. a **video-prior VLM** supplying the only available definition of "moves right",
3. an **optimiser** moving θ toward what the judge prefers.

The deliverable is **plausible parameters**, not recovered true ones. "Did we recover the
true friction" is not a question this project can ask; "does it move like what it is" is.

### The judge is a LIKELIHOOD, and we have been using the wrong function of it

An earlier draft here claimed the judge supplies a prior because there is nothing to
condition on. That was wrong. The judge's approval **is** the observation:

    p(θ | yes)  ∝  p(yes | θ) · p(θ)

`p(yes | θ)` is a proper likelihood — the probability that the judge approves the clip
rendered from θ. MAP is then likelihood times prior in the ordinary way, and weak
identifiability is a property of a flat likelihood rather than something exotic.

**But the logit margin is not the log-likelihood.** With only yes/no in play,

    s        = log p(yes) − log p(no)          ← what we optimise
    p(yes|θ) = sigmoid(s)
    log p(yes|θ) = log sigmoid(s) = −log(1 + e^(−s))    ← the actual log-likelihood

Both are monotone in `s`, so they share an argmax. **Their gradients do not match**, and for
a gradient method that is what counts:

    d/ds  of  s                = 1.000  — constant, never saturates
    d/ds  of  log sigmoid(s)   = 1 − sigmoid(s)
                                 0.500 at s=0, 0.119 at s=+2, 0.047 at s=+3, 0.007 at s=+5

Measured over the 72 scores in the joint fit: median `s` = +1.79, so median `p(yes)` =
**0.857**, and **39% of evaluations sat above p(yes) = 0.90**. At the median the true
log-likelihood gradient is 0.143 — **7× smaller than the quantity we were ascending**, and
falling fast with `s`.

This is a live candidate for the coin-flip gradients. In the saturated region the
likelihood is nearly flat — the judge already says yes to everything we show it — so
differences in `s` there are dominated by whatever else moves the logit (appearance, motion
magnitude) rather than by physics. We were climbing a surface the proper objective says is
already level.

Two consequences:
1. **Optimise `log sigmoid(s)`, not `s`.** It saturates where the judge is confident, which
   is the correct behaviour and stops the optimiser chasing noise in a region it has already
   satisfied.
2. **Keep the fit out of saturation.** If every candidate scores p(yes) > 0.9, the
   experiment carries almost no information. Candidates should straddle the decision
   boundary — which argues for deliberately including implausible θ, the way the authored
   violations did, so the judge is not always answering yes.

## What we measured (not assumed)

**The judge's capabilities are split, and the split is sharp.**

| capability | evidence |
|---|---|
| name object and material | 4/4 correct, open-ended: "a rubber duck, made of rubber"; "a large, metallic pot with a lid" |
| judge motion detail | bounce counts answered 1–2 against a true range of 1–21; ρ(model, true) = −0.23 at 12 frames, +0.30 given all frames |
| rank by motion magnitude | ρ up to −0.88 across 240 evaluations; +1.000 across authored-violation conditions |
| detect gross violations (Super) | 9/9 vs Nano's 1/9 on permanence, shape constancy, teleportation |
| discriminate restitution | ρ = 0.000 against material prior, 4 objects × 3 seeds |

**Probe geometry decides what is measurable.**
- `base/height ≥ 1.4` slides → slip angle reads friction
- `≤ 0.9` topples → topple angle reads centre of mass
- spheres roll → no slip threshold at all (corr +0.32 vs +0.92)

**Threshold observables beat magnitude observables.** Every magnitude-type observable tested
was confounded with "how much it moves". The only off-chance gradient in the project came
from the one threshold observable:

| probe | observable | type | sign consistency |
|---|---|---|---|
| tilt | slip angle | threshold | **5/6, 4/6** |
| drop | bounce height/count | magnitude | 3/6, 2/6 |
| collide | momentum transfer | magnitude | 3/6, 4/6 |

**Degeneracies are structural.** Contact depends only on `k/m` and `cd/m`, so a drop cannot
separate damping from mass. Friction and topple angles are mass-independent. Correcting
density 600 → real shifts restitution by ≤ 0.029 in e, because Hunt–Crossley damping is
`cd·pen·v` and penetration grows with mass, partly self-compensating.

**Instrumentation errors dominated four earlier conclusions.** The judge received 4 frames
regardless of input (`do_sample_frames` default); the bf16 head quantised the score to
multiples of 0.125; a tight crop removed the incline the sliding was judged against; the
floor plane occluded the environment's ground. Each was found by looking at what the model
actually received, not by reasoning about it.

## How to improve, in order

### 1. Weight probes by measured consistency — free
`w_p` already exists and defaults to 1. Tilt is 5/6 with |dy| 0.40; collide is 3/6 with
|dy| 1.72. Unweighted summing lets the least reliable term drive every step. Set weights
from sign consistency.

### 2. Fix the collide probe — it has a concrete failure
ρ = 244 kg/m³ recovered for brass (~8500). The probe that exists to pin density is the one
that failed. Hypothesis: at 1.5 m/s the transfer is over in fewer frames than the judge
resolves. Test: slower impact, more frames on contact.

### 3. Held-out validation — the missing check
Every result comes from **one HDRI, one table, seven staged assets**. Nothing tells us
whether the tilt probe's 5/6 survives a different environment or an unseen object. Fit on a
subset, test on held-out objects and held-out HDRIs. If consistency collapses, we fitted the
subset rather than the physics.

### 4. Average over appearance — regularisation, not more data
The judge is measurably appearance-sensitive: EEVEE rendering the brass pot as matte plastic
inverted its preference. Fitting under one environment risks fitting θ to that appearance.
Rendering each θ under 2–3 environments and averaging removes appearance as a nuisance the
same way SPSA's `y⁺ − y⁻` removes per-object bias.

### 5. More threshold probes — the validated principle

| probe | observable | reads | why it is threshold-type |
|---|---|---|---|
| topple on ramp | angle at which it tips | centre of mass | **free** — the tilt scene already produces it for three objects and we discard it |
| stack and tilt | angle at which a stack falls | CoM + friction jointly | a stack falls at one angle, not gradually |
| rocking period | oscillations per second of a bowl settling | CoM height, inertia | **period is amplitude-independent** for small oscillations — magnitude-invariant by physics, not by design |
| roll-vs-slip transition | angle where a sphere switches from sliding to rolling | friction, for spheres | recovers the objects currently excluded |
| drape | fold shape at rest | cloth stiffness | static — magnitude cannot carry it |
| press–release | deformation shape at matched displacement | Young's modulus | needs a soft-body solver |

The rocking-period idea is the strongest new one: frequency observables are magnitude-
invariant *as physics*, not as an experimental trick, which is a stronger guarantee than
matching slide distances by construction.

### 6. More objects — after, not before
5 assets and 40 gradient steps gave 45% sign consistency. More objects measured chance more
precisely. They become valuable once (1)–(4) produce a signal to generalise, and as held-out
tests under (3) — which is a different use from adding training data.

## What is built and reusable

Newton sim with Hunt–Crossley contact and CoM offset; Cycles render harness with resume,
shadow-catcher look, multi-object scenes and per-frame tilt; per-object cropping with a
60%-of-frame floor; motion-budget guard; absolute and pairwise judges with fp32 head and
full-frame decode; int8 reasoner path fitting a 33B Cosmos 3 on one 48 GB card; differentiable
preprocessing verified to 5.9e-08; CEM; SPSA with object groups and joint multi-probe
objectives.

The machinery is not the bottleneck. The signal is.
