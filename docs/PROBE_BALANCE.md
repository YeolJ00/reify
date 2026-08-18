# Pan balance — working

## Why this probe

Mass has never been observable in this rig. `collide` was built for it and measures nothing:
a body shoved at fixed velocity slides `v²/(2μg)`, independent of mass. A balance fixes that —
the beam tips toward the larger moment, so the observable is the **sign** of
`m_obj·d_obj − m_ref·d_ref`, a threshold rather than a magnitude, and therefore immune to the
motion-magnitude confound that has tracked every other reading here (ρ up to −0.88).

## Result

Fixed 0.30 kg unknown, sweeping the reference. **7/7**, and the sign flip brackets the unknown
at **[0.26, 0.34] kg against a true 0.30**. The balanced case settles at −0.35°, i.e. level.

| m_ref | expected | settled tilt |
|---|---|---|
| 0.10 | object down | +9.00° |
| 0.20 | object down | +9.00° |
| 0.26 | object down | +9.00° |
| 0.30 | balance | −0.35° |
| 0.34 | reference down | −9.00° |
| 0.40 | reference down | −9.00° |
| 0.60 | reference down | −9.00° |

## The rig

**Exact revolute constraint** (`mesh_contact.apply_revolute`, `MeshProbeScene.set_hinge`).
Reduced-coordinate: position pinned to the anchor, rotation projected to the twist about the
axis by swing-twist decomposition, linear velocity zeroed, angular velocity projected onto the
axis. No stiffness parameter, so nothing to tune and no compliance biasing the threshold.
Verified in isolation — a hinged beam spun at 2.0 rad/s under zero torque holds `wy = 2.0000`
with off-axis components exactly 0.

Damping and angle limits are stored **per body**: a scene holds a beam that swings and settles
alongside a stand that must never move, and a single shared value let whichever hinge was
configured last silently overwrite the other.

**Angle stops at ±9°.** A real balance swings a few degrees against stops that keep its pans
near level. Without them the pans tipped with the beam and their contents slid out over the
rim — measured, both weights ended on the *table*, so the reading came from the moment before
they escaped rather than from a weighing. With stops, heavier pivot damping (so the beam
arrives at the stop rather than slamming into it), a deeper pan rim and higher pan friction,
**6/6 weights stay in their pans** for the whole clip.

**A centre column and foot.** Structural, and a real body in the sim rather than a render-only
prop so that anything falling on it collides properly. It ends 5 cm below the pivot: the beam
underside at tilt *t* and radius *x* sits at `pivot_z − (BEAM_T/2)cos t − x·sin t`, so a column
reaching to just under the level beam is struck as soon as it tilts.

## Errors worth remembering

- **Coarsening the timestep to speed a render changed the answer**: 7/7 → 4/7 when substeps
  went 60 → 24 and frames 48 → 40. The physics was never the problem.
- **The stand was buried to its centroid.** `MeshProbeScene` COM-centres every body, so
  placing it at `GZ` sinks it; placement goes through `rest_height()`.
- **The render prep hard-coded three body names** while the scene had four, silently shifting
  every body one slot — the stand rendered as a weight box and the reference mass vanished
  from the scene. It now asserts the count matches.
- Three earlier modelling errors were each real and each fixed nothing on their own (pans as
  free bodies sliding off the beam, a frictionless undamped pivot sampling arbitrary pendulum
  phase, weights placed by centre at the pan surface and so starting buried in it). The fault
  was upstream: `calibrate_stiffness` sizes `k` from body length, so a 0.60 m beam got a soft
  spring, and the pair contact combines springs in series — giving ~18 mm of penetration
  against an 8 mm pan floor, so every weight sank through its pan onto the table.
