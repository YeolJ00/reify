# Pan balance — mass of a real asset

Mass is not observable from sliding: a body shoved at fixed velocity travels `v²/(2μg)`,
independent of its mass, which is why the `collide` probe measured nothing. A balance tips
toward the larger moment, so the observable is the **sign** of `m_obj·d_obj − m_ref·d_ref` —
a threshold, immune to the motion-magnitude confound.

## Result

Weighing **`food_apple_01`, true mass 0.182 kg**, against calibrated reference blocks.

| m_ref | expected | settled tilt |
|---|---|---|
| 0.05 | object down | +12.00° |
| 0.10 | object down | +12.00° |
| 0.15 | object down | +12.00° |
| 0.18 | object down | **+6.36°** |
| 0.22 | reference down | −12.00° |
| 0.30 | reference down | −12.00° |
| 0.50 | reference down | −12.00° |

**7/7.** The sign flip brackets the unknown at **[0.18, 0.22] kg against a true 0.182**, and
the near-matched case settles at +6.36° rather than saturating — the tilt carries magnitude
near balance, not just a sign.

**14/14 payloads stay in their pans**, settling 0.2–3.2 cm from the pan axis in a 14.2 cm bowl.

## The two things that made it an instrument

**The pivot sits 30 mm above the beam's centre of mass.** `MeshProbeScene` puts a body's origin
at its vertex mean — essentially its CoM — so pinning the origin made the balance pivot through
its own centre of mass. That is *neutral equilibrium*: any imbalance swings it straight to the
stop and the tilt angle carries no magnitude at all, which is why every case saturated at ±12°.
`set_hinge(pivot_local=...)` now takes a body-local pivot point. The rise above the CoM is the
balance's sensitivity — larger is stiffer and less sensitive.

**The pans are real scanned bowls.** A flat pan lets a round subject roll to the rim and climb
out; the apple left the pan entirely at 12° of tilt. Every fabricated alternative failed
differently: a stepped well ejected the subject on contact with its shelf, a snug pan jammed
it, and hanging pans swung it out. A bowl is concave all the way to its centre, so a round
object settles at the bottom and stays. This is why real balance pans are dished.

## Errors worth remembering

- **Payloads were seated on the bowl's *outer* bottom.** A bowl is a shell: near its axis the
  lowest vertices are the outside of the base and the highest are the inner surface. The
  subject started 12 mm inside the floor and the contact ejected it. `pan_inner_z()` measures
  the inner surface instead of assuming a flat plate.
- **The body origin is not the beam's centre.** Adding two large bowls moves the assembly's
  vertex mean well above the beam box, so every height derived from "the beam is at z_beam"
  was wrong. Work back through `s.coms[0]`.
- **Cylindrical references roll.** One walked 69 mm inward, shortening its own moment arm by
  23% — an error in exactly the quantity being compared. References are blocks.
- **Coarsening the timestep to speed a render changed the answer**: 7/7 → 4/7 at 24 substeps
  instead of 60.
- **`calibrate_stiffness` sizes `k` from body length**, so a long beam gets a soft spring; in
  series with a payload that gave ~18 mm of penetration against an 8 mm pan floor and every
  weight sank through its pan onto the table.

## Hanging pans: tried, abandoned, kept in the codebase

`MeshProbeScene.add_link` suspends one body from another by a point-to-point spring, attached
above the hung body's centre of mass so gravity levels it. It is the more realistic mechanism
and it works as a joint — but a suspended pan is a pendulum, and its swing threw the payload
out (sweeps went 7/7 → 5/7 → 2/7). Welded pans retain the contents. The link is kept because
it is the only way to express "hangs from a point that is itself moving": `apply_revolute`
pins to a fixed world anchor, and a hard projection would teleport the pan without the beam
ever feeling its weight — and that load is the whole measurement.
