# Pan balance — built, NOT working (2/7)

## Why this probe

Mass has never been observable in this rig. `collide` was built for it and measures nothing:
a body shoved at fixed velocity slides `v²/(2μg)`, independent of mass. A balance should fix
that — the beam tips toward the larger moment, so the observable is the **sign** of
`m_obj·d_obj − m_ref·d_ref`, a threshold rather than a magnitude.

## What was built and does work

**An exact revolute constraint** (`src/sim/mesh_contact.py::apply_revolute`,
`MeshProbeScene.set_hinge`). Reduced-coordinate: after each integration step the body's
position is pinned to the anchor, its rotation is projected to the twist about the joint axis
via swing-twist decomposition, linear velocity is zeroed and angular velocity is projected
onto the axis. No stiffness parameter anywhere, so nothing to tune and no compliance to bias
the threshold. Optional exponential pivot damping specified as a rate per second, converted to
a per-step factor so settling time is independent of `dt`.

**Procedural bodies in `MeshProbeScene`** — passing a `trimesh.Trimesh` in place of an asset
name now works, so rig parts share the same exact mesh contact as scanned assets.

## What does not work

Sweeping the reference mass against a fixed 0.30 kg unknown:

| m_ref | expected | settled tilt | correct |
|---|---|---|---|
| 0.10 | obj down | −7.22° | no |
| 0.20 | obj down | −0.43° | no |
| 0.26 | obj down | −1.77° | no |
| 0.30 | balance | +0.08° | yes |
| 0.34 | ref down | +0.32° | no |
| 0.40 | ref down | −3.79° | yes |
| 0.60 | ref down | +9.29° | no |

**2/7.** The tilt does not track the moment difference — it is not merely noisy, the sign is
wrong in both directions and the magnitudes do not grow with the imbalance.

## Three modelling errors found and fixed along the way (none of which fixed it)

1. **Pans were free bodies resting on the beam**, so they slid off as soon as it tilted and
   the tilt reported where the pans went. Beam and pans are now one rigid body with a rim on
   each pan. 2/7 → 3/7.
2. **The pivot was frictionless and undamped**, so the beam swings as an undamped pendulum
   and any fixed frame samples an arbitrary phase rather than the equilibrium. Added pivot
   damping and read the mean over the final quarter. 3/7 → 1/7.
3. **Weights were placed by their centre at the pan surface**, starting 2.5 cm buried in the
   pan floor. Fixed to sit on it. 1/7 → 2/7.

Each was a real error. None was *the* error, which means the remaining fault is upstream of
all three.

## Where to look next

The prime suspect is the interaction between the hinge projection and contact. `apply_revolute`
runs **after** `integrate_6dof` and overwrites the beam's state unconditionally, discarding
whatever angular impulse the contact forces just delivered along non-axis directions — but it
also discards the axis-aligned impulse if the projection order is wrong relative to how torque
accumulates. A constraint applied as a hard post-hoc projection cannot distinguish "this
velocity violates the joint" from "this velocity is the joint responding to a load".

Worth testing before anything else, cheapest first:
- Drive the beam with a **known torque** (no weights, no contact) and check the angular
  acceleration matches `τ/I` about the axis. That isolates the constraint from the contact.
- Log the beam's angular velocity per step for one imbalanced case and see whether contact
  impulses are surviving the projection at all.
