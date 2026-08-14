# The larger scene — 14 objects, one tilt ramp, one render

Every fit before this used **two** objects (brass_pot, wooden_bowl), because probes were
paired to specific geometry one at a time. The tilt ramp is a *scene-level* experiment:
tilt the table and every object declares itself at once. One rollout, one render, 14
readings.

## What is in it

14 assets, chosen to span probe types rather than to look nice:

| probe type | reads | objects |
|---|---|---|
| slide (`base/h ≥ 1.4`) | friction | wooden bowl, C-clamp, shark |
| topple (`base/h ≤ 0.9`) | centre of mass | ceramic vase, book set, rubber duck, triceratops, lion |
| roll | neither — no slip threshold | baseball, apple |
| mixed | both | brass pot, teapot, cardboard box, file sorter |

## Results, μ ascending

Ramp 0°→26° over 2 s after a 12-frame settle, all 14 bodies in **one** rollout with
per-body friction, so they genuinely collide.

| object | μ | true slip | onset | travel | type |
|---|---|---|---|---|---|
| C-clamp | 0.18 | 10.2° | 13.8° | 38.1 cm | slide |
| wooden bowl | 0.22 | 12.4° | 15.5° | 24.2 cm | slide |
| teapot | 0.24 | 13.5° | 13.8° | 19.0 cm | mixed |
| brass pot | 0.26 | 14.6° | 17.7° | 17.0 cm | mixed |
| shark | 0.30 | 16.7° | 17.7° | 30.3 cm | slide |
| file sorter | 0.38 | 20.8° | 19.9° | 14.4 cm | mixed |
| cardboard box | 0.45 | 24.2° | none | 0.7 cm | mixed |

The sliders order correctly by μ and the box — whose true slip angle is above the ramp's
26° maximum — correctly never moves. Topplers and rollers trip the detector immediately
(0.6–1.7°) because they tip or roll rather than slide; that is the probe-geometry rule
working, not a failure.

## Three bugs this scene exposed

1. **`ProbeScene` had one global μ.** A multi-object tilt ramp is meaningless if every
   object shares a friction coefficient. Friction is now per-body, with the pair
   coefficient the geometric mean `sqrt(mu_i * mu_j)` — symmetric, so contacts still obey
   Newton's third law. Verified: μ=0.18 clamp slid 137.7 cm where a μ=0.55 bowl moved 4.4.

2. **The render read the wrong scene config.** `blender_render_scene.py` always loaded
   `outputs/scene/scene.json` (7 objects, short names) while the poses named 14 assets by
   folder. Nothing matched, so every object was parked 3 m out of frame and the render was
   an empty table. It now prefers `LAB/scene.json` and *raises* on a name mismatch instead
   of silently rendering nothing.

3. **Poses were written in the wrong frame.** The simulator keeps the ground flat and
   rotates gravity; the renderer keeps gravity down and rotates the table. Raw simulator
   poses leave objects on a flat plane while the table rotates out from under them — which
   renders as objects being *crushed flat* as the frames diverge. Poses are now carried
   into the tilted-table frame (rotate about the pivot by the same angle) before export.

## Rendering

Blender 4.4.3 Cycles, city HDRI, shadow-catcher floor. Controls added:

- `RENDER_SCALE` — multiplies output resolution. The judge sees 544×448; a human should
  not have to. Used 3 → **1632×1344**.
- `CAM_PULL` — backs the camera along its sight line for wider scenes, preserving framing.
- `TABLE_SX` / `TABLE_SY` — widen the table in X/Y only, keeping the surface at
  `ground_z`. 14 objects need 0.54 m of depth; the stock table gives 0.706 with no margin.
  At 1.4× the usable surface is 1.588 × 0.988 m.

Videos: `docs/videos/bigscene_tilt_a.mp4` (three-quarter), `bigscene_tilt_c.mp4` (overview).
