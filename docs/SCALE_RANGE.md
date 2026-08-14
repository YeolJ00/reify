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

## The missing constant: a double-scaled sphere cover

`ProbeScene` takes `pitch` in **unscaled mesh units** and scales it internally
(`pitch_b = pitch * sfac`, probe_scene.py:155). The scale-test harness passed
`pitch=PITCH*s` *and* `mesh_scale=[s]`, so the cover was built at `PITCH*s²`.

The body is a cloud of penalty springs, so the number of spheres **is** the contact
stiffness. Double-scaling made it vary by 1800x:

| scale | spheres used | sphere radius | intended |
|---|---|---|---|
| 0.15 | **72,977** | 0.107 mm | 590 |
| 1.00 | 1,571 | 4.77 mm | 590 |
| 6.00 | **40** | 171.6 mm | 590 |

At 0.15x the body rested on ~73,000 springs instead of 590 — roughly 100x too stiff,
which is the launch. At 6x it was 40 balls of 17 cm radius, which is not a bowl.

An earlier note in this file claimed "pitch scales with the mesh, so sphere count stays
constant by design". That was measured from the harness's own `sphere_cover` call, not
from the cover the simulator actually used. The two differed by the whole bug.

**Every production call site (`joint_sim.py`, `j2r_calibration.py`, `multiprobe_build.py`,
`spsa_*.py`, ...) passes `pitch=PITCH` correctly.** The defect was confined to the
scale-test harness; no fitted result is affected.

## Scale after the fix

Correct cover + `k ∝ s²`, wooden bowl, μ = 0.40 (true slip angle 21.8°):

| scale | size | before | after |
|---|---|---|---|
| 0.15 | 2.9 cm | 5.1° (broken) | **21.5°** |
| 0.35 | 6.8 cm | 5.1° (broken) | 23.6° |
| 1.00 | 19.3 cm | 22.5° | 26.9° |
| 2.50 | 48.3 cm | 29.1° | 28.0° |
| 6.00 | 116.0 cm | 32.4° | 32.4° |

**Spread 27.3° → 10.9°; mean error 10.4° → 4.8°.** The two small scales went from
returning a friction-independent constant to reading 21.5° against a true 21.8°.

`k ∝ s²` only helps once the cover is right — with a correct cover it cuts the spread
from 27.3° to 10.9°; with the broken cover it did almost nothing. That is why four
correct fixes in a row appeared to fail: they were all masked by this one.

Friction is now monotone in μ at every scale, which is what an optimiser needs:

| μ | true | 0.15x | 0.35x | 1.0x | 2.5x | 6.0x |
|---|---|---|---|---|---|---|
| 0.20 | 11.3° | 12.7 | 14.9 | 17.1 | 18.2 | 21.5 |
| 0.40 | 21.8° | 21.5 | 23.6 | 26.9 | 28.0 | 32.4 |
| 0.70 | 35.0° | 29.1 | 33.5 | 36.7 | 34.5 | none |

## Residual: the ramp rate is still fixed

A ~+10° monotone drift from 0.15x to 6x remains, and it has a predicted cause. The ramp
sweeps 4°→40° in a fixed 1.42 s of simulated time at every scale. A body sliding from rest
covers the 5%-of-size threshold in `t ∝ √L`, so **larger objects take longer to trip the
detector and the ramp has advanced further by then** — exactly the observed sign and
monotonicity. The fix is to scale the ramp duration as `√L` (or extrapolate onset to zero
threshold). Not yet implemented.

## Working range

**Roughly 0.15x–2.5x (3–50 cm) for a monotone friction readout**, best near 1x. Beyond
that the residual ramp-rate drift dominates and 6x loses the high-μ case entirely.

The scene is still one fixed 0.706 m table; a 116 cm bowl on it is not a sensible
experiment regardless of solver behaviour.
