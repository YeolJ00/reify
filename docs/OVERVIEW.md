# Making authored assets simulation-ready — what we tried, and why the design is what it is

## The problem

An authored 3D asset has geometry and appearance but no physics. To drop it into a simulation
you need **θ** — mass, friction, restitution, centre of mass, stiffness. Those are not in the
file, and unlike a real object there is no ground truth to measure: the asset never existed, so
there is no footage of it falling over.

That single fact drives every design decision below.

---

## Act I — Video generation as a witness. Abandoned.

**What we did.** Give a video model the asset's first frame and a prompt, let it generate the
motion, track points through the generated footage, and fit θ so the simulator reproduces those
tracks. If the model has a good physical prior, its video is a substitute for the missing
footage. We ran this properly across two backends and fifteen seeds, with a full
track-and-fit pipeline behind it.

**What we measured.**

| backend | runs | track survival | recovered `gravity_z` | recovered `tri_ke` |
|---|---|---|---|---|
| Wan 5B | 6 | 62% | −1.84 … −0.62 | 4 913 … 55 553 |
| HunyuanVideo | 9 | 66% | **−6.12 … +5.97** | 3 803 … 46 041 |

True `gravity_z` is **−9.81**. Configured `tri_ke` is **5 000**.

**Why we moved on.** Not because the fit was noisy — because the footage is not physical.
Wan's implied gravity is a sixth of the real value; Hunyuan's spans zero and comes out
**positive**, meaning the fit's best explanation of the generated video is that things fall
upward. A third of the tracked points do not survive the clip at all, because the generated
objects do not maintain identity frame to frame.

The failure is structural, not a tuning problem. **If you fit to generated pixels, hallucination
enters θ directly**, and there is no ground truth to catch it. A model that renders a plausible
*looking* video under an impossible physics will hand you impossible parameters and no
indication that anything went wrong.

---

## Act II — The reformulation: the simulator generates, the model judges

Invert the roles.

- The **simulator** (Newton/Warp) proposes candidate motions under different θ. Every candidate
  is physically valid *by construction* — it came out of a physics engine, so no impossible
  motion can ever be proposed.
- The **video model** (Cosmos 3) never generates. It only *ranks* clips we manufactured.
- The **optimiser** moves θ toward what the judge prefers.

This makes hallucination harmless. The judge can be wrong about which clip is better, and the
worst case is a bad θ *within the space of physically realisable θ* — it can never produce
upward gravity, because no such rollout was ever offered to it.

The corresponding rule, which the whole codebase is built around: **the video model is a judge,
never a witness.**

---

## Why SPSA

The chain is `θ → simulation → render → judge → score`. Two links are not differentiable in
practice: Cycles path-tracing is not, and backpropagating through a 33 B parameter VLM
quantised to int8 does not fit in 48 GB alongside the renderer. So gradients have to come from
finite differences — and the naïve version costs `2 × dim(θ)` evaluations per step, where each
evaluation is a simulation, a Blender render and a VLM forward pass.

**SPSA costs 2 evaluations per step regardless of dimension.** It perturbs *every* coordinate
at once with a random sign vector and divides the resulting scalar difference back out. Going
from 1 parameter to 3 — or to 3 per object across 14 objects — is free. The cost scales with
the number of *probe scenes*, not with the number of parameters.

### The formulation

Each object *o* carries its own parameter vector, in log space so the optimiser takes
multiplicative steps and cannot propose a negative friction:

```
θ_o = ( log₁₀ μ ,  log₁₀ cd ,  log₁₀ ρ )        θ_o(0) = log₁₀(0.40, 2500, 1200)
```

The objective is the judge's approval, summed over probes:

```
J(θ) = Σ_p  w_p · log σ( s_p(θ) )
```

where `s_p` is the judge's logit margin on probe *p*'s clip. **`log σ(s)`, not `s`.** Both are
monotone in the margin so they share an argmax, but their gradients differ: `d/ds` of the raw
margin is a constant 1, while `d/ds log σ(s)` is `1 − σ(s)`, which falls to 0.12 at `s = +2`.
The measured median margin was `+1.79` with 39% of evaluations above `p(yes) = 0.90` — deep in
saturation, where the raw margin keeps reporting a gradient the true likelihood says is flat.

One SPSA step, with gains on Spall's schedule (`α = 0.602`, `γ = 0.101`):

```
a_k = 0.40 / (0.1·N + k + 1)^α          c_k = 0.14 / (k + 1)^γ

Δ_o ~ Rademacher(±1)³                    drawn independently per object
θ_o^±  = θ_o ± c_k Δ_o

ĝ_o = Σ_p  w_p ⊙ m_p ⊙ [ log σ(s_p^+) − log σ(s_p^−) ] / ( 2 c_k Δ_o )

θ_o ← clip( θ_o + a_k ĝ_o ,  bounds )    ascent on log-likelihood
```

Three details carry real weight:

**`m_p` — the probe mask.** A 0/1 gate confining each probe's gradient to the coordinates it
can actually see. Without it, the loudest term steers everything: the retired `collide` probe
had the largest `|dy|` (1.72) and the worst sign consistency (3/6), and it was pushing μ, about
which it knows nothing.

**`w_p` — measured, not assumed.** Fisher information for a yes/no judge is `p(1−p)·(∂s/∂θ)²`,
so both saturation and insensitivity cost. Tilt was the *most* sign-consistent probe (5/6) and
the *least* informative, because its mean `p(yes)` is 0.948 and `p(1−p) = 0.047`. Consistency
and informativeness are different properties, and equal weighting confuses them.

**Both signs share a scene.** `θ⁺` and `θ⁻` are the same object rendered in the same setup, so
any per-object bias in the judge — it likes brass pots, it dislikes clutter — cancels in
`s⁺ − s⁻`. This is why SPSA is a better fit here than a parameter sweep: a sweep compares
across objects, where shape dominates every observable we have measured.

---

## Act III — Building the instrument

With the roles inverted, the question becomes: **what experiment makes θ visible at all?**

This is where most of the work went, and where the governing rule came from a failure. The
`collide` probe was built to recover mass by shoving an object and measuring how far it slid.
It measures nothing: a body pushed at fixed velocity decelerates at `μg` and travels
`v²/(2μg)` **regardless of its mass**. No amount of tuning could have fixed it, because the
observable does not depend on the parameter.

Hence: **a probe must be shown to depend on its parameter before it is trusted.** Sweep the
parameter, report the correlation or the threshold shift, *then* report a recovery.

Six probes now pass that test:

| probe | observable | reads | result |
|---|---|---|---|
| tilt | slip angle | friction vs table | 0.40° residual, 3 cm – 116 cm |
| balance | which pan drops | **mass** | 7/7, brackets 0.182 kg at [0.18, 0.22] |
| bounce | rebound height | contact damping | ρ = −0.995 |
| settle | decay of rocking | contact damping | ρ = +0.973 |
| spin | spin-down time | friction vs table | ρ = −0.952 |
| stack | collapse angle | friction **between objects** | ρ = +0.994, residual 2.41° |

Two do not: cloth drape reads stiffness only weakly (ρ = −0.786, saturated at the soft end
because a 7×5 grid cannot represent a fold), and the soft-body press diverges.

### A result that changed the plan

The project's standing rule was that only *threshold* observables work — slip angles, sign
flips — because every *magnitude* observable had been confounded with "how much it moves"
(ρ up to −0.88 with motion magnitude). A drop probe measuring rebound height gave ρ = −0.186
and seemed to confirm it.

That −0.186 was measured **across 14 different objects**, where shape dominates rebound. Holding
the object and the excitation fixed and sweeping only the damping, the *same observable* gives
**−0.995**.

The confound was never that magnitudes are bad. It was comparing magnitudes across objects.
Magnitudes are fine *within* an object — which is exactly the regime that per-object θ puts us
in, and it re-opened half the probe design space.

---

## Where it actually stands

The simulator side is strong. Contact is exact against the mesh — vertices against the plane
for the ground, Warp's BVH in the other body's local frame for body–body — after a sphere-cover
approximation was found to be 2.28 mm from the true surface. Friction, mass, damping and
contact stiffness are per-body. The table is an object with its own material. Scale invariance
holds from 3 cm to 116 cm.

**The judge is the bottleneck, and it has not moved.** It approves 52.8% of clips at
`p > 0.90`, and correlates −0.02 with density, −0.07 with damping and −0.25 with friction
across a 10× parameter sweep. It names materials 4/4 correctly but answers bounce counts of
1–2 against a true range of 1–21. Three fits of the same object returned densities of 244, 188
and 356.

We can now manufacture excellent evidence and still cannot read it off a video.

What has changed is *what the judge is being asked*. Several probes are now one-bit thresholds
— **which pan went down**, **did it slide at this angle** — rather than magnitude comparisons.
That is a question a VLM plausibly answers well, unlike counting bounces, and the balance gives
ground truth to score it against. That test is cheap and decisive, and it is the next thing to
run.
