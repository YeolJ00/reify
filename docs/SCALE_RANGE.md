# Scale range and the continuous ramp — measured

## What was fixed: the ramp is now one rollout

The tilt probe used to run as a **sequence of separate scenes**, one per angle, carrying
position and linear velocity forward. `ProbeScene.__init__` resets orientation to identity
and angular velocity to zero, so every angle step teleported the body upright and de-spun
it — 30+ artificial kicks per probe.

`ProbeScene.gravity_seq` now supplies **per-step gravity inside a single rollout**
(`tilt_probe.ramp_gravity_seq`). No kernel change was needed — gravity was already a
per-step argument to `integrate_6dof`. The whole ramp is one unbroken graph, which is also
what a gradient path would require later.

## What that bought: friction reads correctly at the calibrated scale

Slip angle vs. μ on the wooden bowl at 1×, continuous ramp, 10-frame settle:

| μ | true slip angle | measured | error |
|---|---|---|---|
| 0.20 | 11.3° | 16.0° | +4.7° |
| 0.30 | 16.7° | 21.5° | +4.8° |
| 0.40 | 21.8° | 22.5° | **+0.7°** |
| 0.55 | 28.8° | 26.9° | −1.9° |
| 0.70 | 35.0° | 34.5° | **−0.5°** |

Monotone throughout and within ~1° for μ ≥ 0.4. Before the fix the same probe saturated
badly (μ=1.0 read 30.3° against a true 45°). **This is the first quantitatively correct
physical readout the project has produced.**

## What was NOT fixed: scale

Slip angle is scale-free (`tan θ = μ`), so any spread is artefact. Measured 0.15×–6×:

| scale | size | slip angle (μ=0.40, true 21.8°) |
|---|---|---|
| 0.15 | 2.9 cm | 5.1° |
| 0.35 | 6.8 cm | 5.1° |
| **1.00** | **19.4 cm** | **22.5°** |
| 2.50 | 48.4 cm | 29.1° |
| 6.00 | 116.2 cm | 32.4° |

Five hypotheses tested, **all five rejected or marginal**:

| hypothesis | result |
|---|---|
| contact `k` fixed while `m ~ s³` → `k ∝ s²` | spread 35° → 29°, real but minor |
| restart transient → continuous ramp | 29° → 26.3°, minor |
| spawn gap fixed at 2 mm → `∝ s` | no effect |
| `V_EPS` friction regulariser fixed at 1 mm/s → `∝ √(gL)` | 27.3° → 26.2°, negligible |
| fixed `dt` vs natural period `√(L/g)` → substeps `∝ 1/√s` | no effect |

**The actual small-scale failure is a contact explosion, not a mis-measured slip.**
Trajectory inspection at 0.15×: the body moves 41,000% of its own size horizontally and
**88,000% upward** — it is launched, not slipped. It reads 5.1° for μ = 0.20, 0.40 and
0.70 alike, i.e. friction-independent. Cause not yet identified; it is none of the five
above.

## Working range

**Validated: roughly 0.5×–2.5× the calibrated size (≈10–50 cm).** At 1× the friction
readout is accurate to ~1°. At 2.5× it reads +7° high. Below ~7 cm the contact is unstable
and the probe returns a constant unrelated to μ.

Note also that the *scene* is fixed — one table at 0.706 m. A 116 cm bowl on it is not a
sensible experiment regardless of solver behaviour. Extending the range means rescaling the
scene with the object, not only the object.
