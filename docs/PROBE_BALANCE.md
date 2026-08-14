# Pan balance — working

## Why this probe

Mass has never been observable in this rig. `collide` was built for it and measures nothing:
a body shoved at fixed velocity slides `v²/(2μg)`, independent of mass. A balance fixes that —
the beam tips toward the larger moment, so the observable is the **sign** of
`m_obj·d_obj − m_ref·d_ref`, a threshold rather than a magnitude, and therefore immune to the
motion-magnitude confound that has tracked every other reading here (ρ up to −0.88).

## Result

Fixed 0.30 kg unknown, sweeping the reference:

| m_ref | expected | settled tilt | correct |
|---|---|---|---|
| 0.10 | obj down | +19.67° | yes |
| 0.20 | obj down | +17.50° | yes |
| 0.26 | obj down | −19.54° | no |
| 0.30 | balance | +2.25° | yes |
| 0.34 | ref down | −19.76° | yes |
| 0.40 | ref down | −19.83° | yes |
| 0.60 | ref down | −18.83° | yes |

**6/7**, and the sign flip brackets the unknown at **[0.26, 0.30] kg against a true 0.30**.
Tilt saturates near ±19.7° because the beam swings to a mechanical stop, which makes the
readout cleanly binary. The one miss is the 13%-imbalance case, where saturation leaves little
margin.

## The mechanism

**Exact revolute constraint** (`src/sim/mesh_contact.py::apply_revolute`,
`MeshProbeScene.set_hinge`). Reduced-coordinate: position pinned to the anchor, rotation
projected to the twist about the axis by swing-twist decomposition, linear velocity zeroed,
angular velocity projected onto the axis. No stiffness parameter, so nothing to tune and no
compliance to bias the threshold. Verified in isolation — a hinged beam spun at 2.0 rad/s with
zero torque holds `wy = 2.0000` exactly, with off-axis components exactly 0.

**Beam and pans are one rigid body** with a rim on each pan. As separate bodies the pans slid
off the moment the beam tilted, and the tilt reported where the pans went.

**Pivot damping**, given as a rate per second and converted to a per-step factor so settling
time is independent of `dt`. An ideal frictionless hinge never settles, so any fixed frame
samples an arbitrary phase of an undamped pendulum.

## The bug that mattered

Everything above was in place at 2/7. The actual fault: `calibrate_stiffness` sizes `k` from
the body's own length, so a 0.60 m beam got a soft spring, and the pair contact uses springs in
series (`kij = ki·kj/(ki+kj)`). That gave ~18 mm of penetration against an 8 mm pan floor —
**every weight sank through its pan and landed on the table**. The trace is unambiguous: weights
settling at z = 0.731, which is table height plus half a box.

Fixed by making the beam stiff (`k = 2e5`, it is a rigid balance beam) and thickening the pan
floor to 22 mm. 2/7 → 6/7.

The lesson matches the sphere-cover one: three plausible modelling errors were found and fixed
first (pans sliding off, undamped pivot, weights buried in the pan floor) and none moved the
result, because the real fault was a scale mismatch upstream of all of them.
