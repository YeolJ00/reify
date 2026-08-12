# Joint multi-probe fit — design

Single-scene fitting is the weakest available configuration: one probe constrains one
direction in theta-space, so fitting one parameter against one scene leaves everything else
unconstrained and leaves that one parameter degenerate with whatever it shares a signature
with.

## Probe -> parameter, and the degeneracies

| probe | constrains | degenerate with | scenes |
|---|---|---|---|
| tilt (ramp) | `mu` from slip angle; `CoM` from topple angle and direction | nothing — both are thresholds, and both are mass-independent | 1 |
| drop | `cd/m` from bounce count and height | **mass** — contact force is `max(k*pen - cd*pen*v, 0)` and acceleration is `F/m`, so only the ratios `k/m` and `cd/m` are identifiable | 1 |
| collide | mass **ratio** from momentum transfer | nothing — this is what breaks the drop's degeneracy | 1, two bodies |
| drape | cloth stiffness from fold shape at rest | nothing — static observable, magnitude cannot carry it | 1 |

Measured, sim-side, no judge involved:
- friction is identifiable from slip angle for flat-based bodies: corr(mu, tan slip) = +0.918 (book on a fixed incline), +0.906 (brass pot)
- a sphere has no slip threshold: baseball corr = +0.319, saturating near 10.5 deg for every mu
- a tall narrow body topples instead of sliding: the ceramic vase moves at 8.5 deg despite mu=0.80
- correcting density 600 -> real shifts restitution by at most 0.029 in e, because Hunt-Crossley
  damping is `cd*pen*v` and penetration grows with mass, partly self-compensating

## Why joint rather than sequential

Mass is the only real dependency: `cd` cannot be read from a drop until mass is pinned. That
argues for sequential fitting — collide first, then drop. But joint fitting handles it
without ordering, because the collision term constrains mass *while* the drop term
constrains `cd/m`, and the two together determine both. Sequential fitting is only necessary
if a probe cannot be run until another has produced a number, which is not the case here.

## Cost

SPSA needs **2 evaluations per step regardless of theta dimension**. One evaluation is one
render per probe scene. So:

    renders per iteration = 2 x (number of probe scenes)

Adding parameters is free. Adding probes costs renders linearly and adds constraint. Fitting
theta = (mu, cd, mass, CoM) against tilt + drop + collide is 6 renders per iteration, against
4 for the single parameter and single probe fitted today.

## Objective

    J(theta) = sum_p w_p * s_p(theta)  +  log prior(theta)

with `s_p` the judge's logit margin on probe `p`'s clip for that object. Weights `w_p`
default to 1; a probe that the calibration shows carries no signal for a given parameter
should be weighted 0 for it rather than quietly averaged in.

Per-object offsets cancel inside SPSA's `y+ - y-` as they do today, because both evaluations
of a step are the same object in the same scenes.

## Status

Built: tilt scene, drop scene, multi-object rendering, per-object cropping, SPSA with
object groups, judge.

Not built: collide scene (needs two bodies with a controlled initial velocity — ProbeScene
already supports both), drape (needs per-frame cloth mesh export rather than rigid poses),
press-release (no soft-body solver exposed in this Newton build).

Next concrete step: add the collide scene and fit `(mu, cd, mass)` jointly over
tilt + drop + collide, which is the smallest configuration that breaks the mass degeneracy.

## Parameter coverage — what we probe and what we do not

| parameter | probed by | status |
|---|---|---|
| friction `mu` | tilt (slip angle), spin (spin-down rate), shove (stopping) | 3 probes |
| density → mass | collide, collide_heavy, collide_slow, stack | 4 probes, all momentum-based |
| centre of mass | tilt (topple angle and direction), stack | 2 probes |
| restitution via `cd` | drop | 1 probe |
| **Young's modulus — squish** | — | **no soft-body solver exposed** (`SolverVBD`/`Style3D` are cloth, `ImplicitMPM` nearest) |
| cloth bend / shear / stretch | — | buildable, `FlagSim` exists and is unwired |
| rolling resistance | — | distinct from sliding friction, unmodelled |
| plastic yield, fracture | — | unmodelled |

**Why mass needs several probes rather than several kinds of probe.** Mass cancels out of
almost every observable available to us:

    spin-down    torque ~ mu*m*g, inertia ~ m*r^2  ->  omega-dot ~ mu*g/r   m cancels
    slip angle   tan(theta) = mu                                            m absent
    topple       geometry of CoM against base of support                     m absent
    free fall    a = g                                                       m absent
    drop bounce  depends on cd/m                                            m only as a ratio

Momentum transfer is the only observable that sees mass on its own, so emphasis means more
collisions: a light partner, a heavy partner, and a slow impact. The slow variant exists
because at 1.5 m/s the transfer finished in fewer frames than the judge resolves, which is
the leading suspect for `rho` returning 188 kg/m³ for brass against a true ~8500.

A balance or seesaw would read mass directly and is the cleanest possible probe for it, but
it needs a pivot — a joint or a static body — and `ProbeScene` integrates free rigid bodies
with penalty contact only. A penalty spring pinning a plank's centre would approximate one
in a few lines and has not been tried.
