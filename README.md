# reify

*re·i·fy* — to make something abstract concrete or real.

An authored 3D asset has geometry and appearance but no physics: no mass, no friction, no
restitution. **reify** recovers those parameters so an asset moves the way its material should.

The hard part is that the asset never existed. There is no footage of it falling over, so
there is nothing to fit against.

## The design, in one paragraph

We tried generating the missing footage with a video model and fitting parameters to it. That
fails structurally — fit to generated pixels and hallucination enters the parameters directly.
Measured over fifteen seeds, recovered gravity ranged from −1.84 to **+5.97** m/s² against a
true −9.81: the best explanation of the generated video was that things fall upward.

So the roles are inverted. The **simulator** proposes candidate motions under different
parameters — every one physically valid by construction. The **video model never generates**;
it only ranks clips we manufactured. The optimiser moves parameters toward what the judge
prefers. Hallucination becomes harmless: the judge can be wrong about which clip is better,
but it cannot invent a motion no physics produces.

> The video model is a judge, never a witness.

Full narrative: [`docs/OVERVIEW.md`](docs/OVERVIEW.md).

## The probes

A parameter is only recoverable if some experiment makes it visible. Each probe here was
validated by sweeping its own parameter *before* any recovery was reported — a rule that came
from a probe which measured nothing at all (`collide` read mass from sliding distance, which is
provably mass-independent).

| probe | observable | reads | result |
|---|---|---|---|
| tilt | slip angle | friction vs table | 0.40° residual, 3 cm – 116 cm |
| balance | which pan drops | **mass** | 7/7, brackets 0.182 kg at [0.18, 0.22] |
| bounce | rebound height | contact damping | ρ = −0.995 |
| settle | decay of rocking | contact damping | ρ = +0.973 |
| spin | spin-down time | friction vs table | ρ = −0.952 |
| stack | collapse angle | friction **between objects** | ρ = +0.994 |
| cloth drape | footprint at rest | stiffness | ρ = −0.786, saturated |
| soft press | compression | Young's modulus | diverges |

## Status

The simulator side is strong: exact mesh contact (vertices against the plane for ground, Warp's
BVH in the other body's local frame for body–body), per-body friction/mass/damping/stiffness,
the table as a parameterised object, scale invariance from 3 cm to 116 cm.

**The judge is the bottleneck.** It approves 52.8% of clips at p > 0.90 and correlates −0.02
with density across a 10× sweep. It names materials 4/4 correctly but answers bounce counts of
1–2 against a true 1–21. We can manufacture excellent evidence and still cannot read it off a
video.

## Layout

```
src/sim/      mesh_contact · mesh_scene · deformable · tilt_probe
src/render/   Cycles harness · crop · camera · views
src/judge/    Cosmos 3 pairwise + absolute
src/optimize/ CEM · SPSA with per-object θ
scripts/      probe_*.py · bigscene_sim.py · joint_fit.py · iter_report.py
docs/         OVERVIEW.md · PROBE_*.md · SCALE_RANGE.md · MESH_CONTACT.md
```

`CLAUDE.md` carries the working rules — each one bought with a failure — and is the file to
read before changing anything.

## Running it

Environment is machine-specific for now: paths to the conda env and Blender are hardcoded in
the scripts. See `CLAUDE.md` for both.

```bash
# a probe, with its parameter sweep
LAB=outputs/scene/balance python scripts/probe_balance.py

# the 14-object tilt scene, then render it
LAB=outputs/scene/bigscene python scripts/bigscene_sim.py assets.json
LAB=$PWD/outputs/scene/bigscene VIEWS=a,c RENDER_SCALE=3 blender -b -P scripts/blender_render_scene.py

# one page per object showing θ per iteration against the clips that moved it
python scripts/iter_report.py outputs/judge/joint --max-clips 4
```

Generated media (`docs/*.html`, `docs/videos/`, `docs/iters/`) is gitignored — all of it is
regenerable from the scripts above.
