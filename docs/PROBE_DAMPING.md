# Bounce and settle — both read damping, and they overturn a prior

Two probes for contact damping `cd`, each swept over a 32× range (400 → 12800) on one object
so shape is held fixed and only the parameter varies.

## Bounce — ρ(log cd, rebound) = −0.995

Dropped 22 cm, nudged 0.22 m/s toward the table edge.

| cd | rebound |
|---|---|
| 400 | 10.14 cm |
| 800 | 8.23 cm |
| 1600 | 6.09 cm |
| 3200 | 4.49 cm |
| 6400 | 3.02 cm |
| 12800 | 1.87 cm |

Monotone across the whole sweep. This is the cleanest parameter reading the project has.

**The threshold version does not work.** "Does it leave the table" transitions between a
0.22 m/s nudge (0/6 leave) and 0.52 m/s (6/6 leave) — it is governed by the nudge, not by
damping, so there is no `cd` threshold to find. Required adding a bounded table
(`MeshProbeScene(table=(half_w, half_d))`) since the ground was an infinite plane and nothing
could fall off an edge at all.

## Settle — ρ(log cd, decay) = +0.973

The bowl is tipped 22° **onto its dome** and released; the decay rate of the rocking is read
from the angular signal's envelope. Upright on its flat foot it does not rock at all — it falls
back and stops — so no period exists; inverted, the curved outside is the contact surface and
it rocks.

Required adding `rot0` to `MeshProbeScene`, which previously hard-coded the initial orientation
to identity. That also made the excitation a controlled **displacement** rather than a velocity
kick, so initial amplitude is now 18.2–19.6° across the whole sweep instead of 7–78°.

**The frequency half does not work.** The dominant FFT bin sits at the lowest non-zero
frequency for every candidate — that is the window length, not a physical period. The motion is
over-damped enough that no oscillation exists to measure, so the amplitude-independence
argument, which was the whole reason to want a frequency observable, cannot be tested here and
is not claimed.

## The prior this overturns

`docs/REFORMULATION.md` records that every magnitude observable tested was confounded with "how
much it moves" (ρ up to −0.88), and concludes threshold observables are the way forward. That
conclusion was drawn from measurements taken **across different objects**, where shape dominates
— an earlier drop probe gave ρ(log cd, rebound) = −0.186 over 14 objects.

Holding the object and the excitation fixed and sweeping only the parameter, the same magnitude
observable gives **−0.995**.

The confound was never that magnitudes are bad. It was comparing magnitudes across objects.
That is a materially different claim, and it means magnitude observables are usable **within**
an object — which is exactly the per-object fitting regime, where each object carries its own θ.
