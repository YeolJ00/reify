# Spin and stack, rebuilt on the mesh solver

Both were in the legacy `joint_fit.py` probe set but were last measured under the sphere
cover, the restart-based ramp and per-scene shared parameters, so none of their numbers
survived. Neither had ever been checked against a parameter sweep, which is the rule that came
out of `collide` — a probe built to recover mass whose observable is provably mass-independent.

## Spin → friction. ρ = −0.952

The bowl is spun at 14 rad/s about its vertical axis and left to grind down. Friction opposes
the spin with a torque of roughly `μ·m·g·r_eff`, so angular deceleration goes as μ.

| μ | μ_eff | time to 1/e of initial spin |
|---|---|---|
| 0.15 | 0.260 | 0.458 s |
| 0.25 | 0.335 | 0.369 s |
| 0.40 | 0.424 | 0.274 s |
| 0.60 | 0.520 | 0.245 s |
| 0.85 | 0.618 | 0.215 s |

Monotone across the sweep. `μ_eff = sqrt(μ_body · μ_table)` because the solver combines contact
coefficients by geometric mean — the body's own μ is not what it slips at.

This is a magnitude observable, but taken **within one object under a fixed excitation**, which
is the regime where the bounce probe measured −0.995. The confound that discredited magnitudes
was comparing them *across* objects, where shape dominates.

## Stack → object-on-object friction. ρ = +0.994

One body on another, table tilted until the stack comes apart. The top breaks away when
`tan θ > μ_pair`, so this is a threshold angle — and it reads the friction **between two
objects**, which no other probe touches: every other contact in this project is against the
table.

| μ_pair | predicted `atan μ` | measured | error |
|---|---|---|---|
| 0.15 | 8.5° | 11.8° | +3.2° |
| 0.25 | 14.0° | 15.7° | +1.7° |
| 0.40 | 21.8° | 23.5° | +1.7° |
| 0.60 | 31.0° | 30.4° | −0.6° |
| 0.85 | 40.4° | 35.3° | −5.1° |

**Bias +0.19°, residual 2.41°.** The μ=0.85 case reads low because at that friction the clamp
begins to tip rather than slide, so the threshold stops being a pure friction reading.

## Three errors, each of which pinned the sweep

1. **Measuring `top − base` in world coordinates.** The base rotates with the ramp by
   construction, so its rotation registered as sliding and every case "collapsed" the moment
   the table moved. The displacement must be taken in the base's own frame. This alone took ρ
   from +0.933 to +0.994 and the residual from 8.98° to 2.41°.
2. **A bowl as the top body.** It rests on a narrow foot ring and rocks off it at ~17°
   whatever the friction, pinning four of five values to the same angle. The top body has to
   slide rather than tip — the C-clamp has base/height 6.3 and lies flat.
3. **A base barely wider than the top.** At scale 1.0 the box top is 24×32 cm and a 19 cm bowl
   ran off the edge almost immediately, so the probe measured the edge rather than the
   friction. The base is scaled to 1.10 and the top is smaller than it.

The ramp also has to run past the largest predicted angle: at a 40° ceiling the μ=0.85 case was
clipped by the sweep itself rather than by physics.
