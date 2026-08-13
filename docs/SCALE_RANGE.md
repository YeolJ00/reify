# Scale range — measured, not assumed

**Question:** does the pipeline work for objects much larger and much smaller than
the ~20 cm assets everything was calibrated on?

**Test:** slip angle is scale-free physics (`tan θ = μ`), so any drift with scale is
an artifact. wooden_bowl, μ = 0.40 (true slip angle 21.8°), scaled 0.15× to 6×.

| scale | size | measured slip angle | with k ∝ s² |
|---|---|---|---|
| 0.15 | 2.9 cm | 4.0° | 4.0° |
| 0.35 | 6.8 cm | 4.0° | 4.0° |
| **1.00** | **19.4 cm** | **24.0°** | **24.0°** |
| 2.50 | 48.4 cm | 29.0° | 13.0° |
| 6.00 | 116.2 cm | 39.0° | 33.0° |

Spread **35°** on a quantity that should not vary at all. Only the calibrated scale
(24.0° vs a true 21.8°) is right.

## Two causes, both implementation

**1. Contact stiffness is fixed while mass scales as s³.** Resting penetration is
`m·g/k`, so pen/size spans 0.013% to 21.5% — a 1300× range. At 6× the bowl sinks
250 mm into a table it should rest on.

Fix: `k ∝ s²` (from `pen = m·g/k` with `m ~ s³`, requiring `pen ~ s`). Verified to
help — spread 35° → 29° — but it is not the dominant term.

**2. The tilt ramp resets orientation every step.** `ProbeScene.__init__` hard-assigns
`rot[0] = identity` (probe_scene.py:235) and defaults `vang0 = 0`. The ramp rebuilds
the scene at each angle carrying position and linear velocity only, so the object is
teleported upright and de-spun 30+ times per probe. The kick is a fixed absolute size,
so it is negligible at 1× and dominant at 0.15×, where it trips the onset detector at
the very first ramp step.

Fix: carry `rot` and `vang` across restarts, or run the ramp as one continuous rollout
with time-varying gravity instead of a sequence of restarts. The second is better —
it also removes the transient at 1× and makes the probe differentiable end to end.

## Where we stand

Validated at the calibrated scale only. Do not treat results from objects far from
~20 cm as physics until the ramp is one continuous rollout. This affects the tilt probe,
which is the project's single best observable (5/6, 4/6 sign consistency) — the restart
transient is present in every one of those numbers too, at 1× where it is small but
not zero.
