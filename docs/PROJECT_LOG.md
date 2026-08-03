# Project log (Claude-facing)

Technical log of everything done, found, and decided. Human-facing narrative:
`docs/report.html` (self-contained, keep both in sync). Date: 2026-07-21.

## Environment

- conda env `warp` → run everything with `/home/jooyeolyun/anaconda3/envs/warp/bin/python`,
  prefix `CUDA_VISIBLE_DEVICES=<free gpu>` (shared 8× RTX A6000 box; check `nvidia-smi`).
- `newton==1.4.0`, `warp-lang==1.15.0`, `opencv-python-headless==5.0.0.93`; pinned in
  `requirements.txt`. PyPI package is `newton` — `newton-physics` is a renamed placeholder.

## Newton 1.4.0 API facts (verified against installed source, not memory)

- `ModelBuilder.add_cloth_grid(...)`: vertex index = `y * (dim_x + 1) + x`;
  `fix_left` pins the `x == 0` column by setting particle mass 0 and clearing
  `ParticleFlags.ACTIVE`.
- `SolverVBD` requires `builder.color()` before `finalize()`.
- `finalize(requires_grad=True)` makes all model arrays differentiable, including
  `model.gravity` which is `wp.array(vec3, shape (1,))` and is read *inside* the
  particle-integration kernel every launch (`_src/solvers/solver.py:88`:
  `v1 = v0 + (f0*inv_mass + gravity*nonzero(inv_mass))*dt`) → gravity is an
  acceleration; internal/wind forces scale by 1/m. Basis of the scale gauge freedom.
- `tri_materials` layout: `(ke, ka, kd, drag, lift)` per triangle;
  `edge_bending_properties`: `(ke, kd)` per edge.
- No built-in wind. We add per-triangle aerodynamic normal pressure
  `f = drag * area * dot(n̂, v_wind − v_tri) * n̂` into `state.particle_f` before each
  substep (`src/sim/rollout.py::wind_force_kernel`, theta version in `theta_sim.py`).
- Determinism: `wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN` must be set
  before solver construction (newton reads it in solver `__init__` and marks its kernel
  modules); our own atomic-emitting modules need explicit
  `wp.set_module_options({"deterministic": ..., "deterministic_max_records": 0}, module=...)`.
  Verified bitwise-identical trajectories across processes. **Cost: 7×** (0.40 s → 2.81 s
  per 1920-substep rollout) → config default `sim.deterministic: false`; enable for
  verification runs only.

## Numerical stability (semi-implicit)

- Explicit damping limit: need `tri_kd * dt / mass ≪ 1`. tri_kd=100 @ mass 0.1, dt 1e-3
  exploded (300 m excursions, finite but garbage). Config keeps ratio ≈ 0.1
  (mass 0.05, tri_kd 10, substeps 32 → dt 5.2e-4).
- Stability cliff in theta space: mass ~10 % below true (at fixed dt) explodes the sim.
  The loss landscape contains this cliff right next to the optimum (visible in
  `outputs/identifiability.png` right panel). CEM tolerates it (NaN sorts last);
  LM rejects non-finite steps.

## Gradient findings

- M1: `d(loss)/d(wind)` through the full rollout via `wp.Tape` matches central FD to
  0.3 % (semi-implicit). **VBD tape gradient is exactly 0.0** in newton 1.4.0 — not
  differentiable; use zeroth-order (CEM/LM-FD) with VBD.
- The 8-dim theta adjoint is correct (dominant components verify to 2–9 % near
  theta_true) but **far from the target the landscape is rugged/chaotic**: FD does not
  converge as h shrinks; the exact local gradient is real but noisy. Plain Adam from a
  far init stalls (~3e-2 plateau); Adam also fails from a CEM-refined init because
  uniform step sizes overshoot stiff directions (condition number 3.3e5).

## theta parameterization (M3)

`theta = [wind_a0, wind_a1, wind_a2, gravity_z, log_tri_ke, log_tri_kd, log_edge_ke,
log_mass]` (`src/sim/theta_sim.py`). Wind speed s(t) = a0 + a1 sin(2πft) + a2 cos(2πft),
f = 1 Hz fixed, direction fixed. Positive params in log space. Fill-kernels write theta
into model arrays *inside the tape* each rollout. YAML gotchas: `true:` key parses as
bool (renamed to `true_values:`); `5.0e3` without sign parses as string (float() casts).

## Identifiability (M3 probe)

Gauss-Newton J^T J from FD trajectory Jacobian at theta_true (PSD; the naive FD
loss-Hessian shows spurious negative eigenvalues because the basin is narrower than the
FD step in stiff directions — don't use it):

- RMS trajectory sensitivity per unit param: log_mass 0.45, log_tri_kd 0.40,
  log_tri_ke 0.083, gravity_z 0.067, wind_a0 0.010, wind_a2 0.0058, wind_a1 0.0054,
  log_edge_ke 0.0017. Condition number 3.3e5.
- Stiffest direction pairs kd against mass (their ratio); wind forcing lives in the
  sloppy subspace; **edge bending is effectively unobservable** in full-trajectory MSE.
- **Scale gauge**: free LM converged to loss 3.3e-8 with mass, tri_ke, tri_kd, wind all
  scaled by the SAME 1.13 factor and gravity exact (0.02 %) — empirical proof that only
  force/mass ratios are observable; gravity (acceleration) is the only absolute anchor
  and it cannot fix the scale.
- **Decision**: gauge-fixed point estimate (pin density/mass from an asset prior) or a
  posterior over scale + bending. `recover_full.py --fix log_mass` recovers all
  observable params to <4 % (wind 0.08/0.99/3.8 %, gravity 0.33 %, tri_ke 0.22 %,
  tri_kd 0.15 %); edge_ke collapses to 0 (unobservable, as predicted).

## Optimizers

- `src/optimize/cem.py`: 1-D and diagonal n-D CEM with decaying additive noise
  (plain CEM collapses sigma by iter ~2 and stalls; noise0 = sigma0/4 annealed to 0
  fixed it). Returns best-ever sample, not final mean (mean drifts along sloppy valley).
- `src/optimize/lm.py`: LM on FD trajectory Jacobian (16 rollouts/iter for 8 params);
  solver-agnostic (works with VBD). Handles the 3e5 conditioning. **Basin lottery**:
  identical runs can land in different minima (GPU atomic nondeterminism alone flips
  basins: 3.3e-8 vs 5.5e-4 across reruns) → `recover_full.py --starts N` multi-start;
  best-of wins. CEM→LM chaining was WORSE than LM from the neutral init (CEM's drift
  in sloppy directions, e.g. edge_ke → 52, poisons the basin).
- Multi-start LM (5 starts, --fix log_mass, 423 s): 3/5 starts reach the global basin
  (final loss ~5e-10); 2 stall in local minima — including the *neutral* config init,
  while perturbed starts won (basin lottery confirmed; multi-start is the fix, and
  ranking by final loss is a reliable selector). Winner recovers **all params to
  ≤0.07 %**, even edge_ke to 1.31 % — at loss 5e-10 the bending signal, ~250× below
  leading sensitivities, finally counts.

## M4 stage 1 (tracked motion from a rendered rollout)

Pipeline (`scripts/run_m4_pipeline.py`): matplotlib rasterizer (pixel-exact, painter
depth sort, static random per-triangle grayscale texture) → chained pyramidal LK with
forward-backward gate (fb < 1 px; = step-8 rejection at tracker level) → tracks attached
to cloth via frame-0 barycentric coords (frontmost triangle wins) → LM on 2D pixel
residuals through `project_bary_points_kernel` (differentiable pinhole, FD-verified to
0.6 %) → gauge-fixed log_mass, multi-start.

- Camera at ~30° azimuth so in-plane and cross-flow motion both project.
- Result (3 starts, 333 s): all starts converge to the same plateau
  ~6e-5 ≈ **4 px RMS — the LK tracking noise floor**, not an optimizer failure.
  Recovery vs true: wind_a0 13.8 %, gravity 19 %, tri_ke 25 %, tri_kd 30 %;
  wind harmonics poor, edge_ke → 0. With exact 3D supervision the same optimizer
  reaches ≤0.07 % — so the accuracy limit has moved from optimization to
  *observation quality*. Levers: better tracker (CoTracker), sub-pixel refinement,
  multi-view, longer horizon, robust/weighted 2D loss.
- Visual check: recovered-sim reprojections lie on top of LK tracks across the whole
  deformed flag (`outputs/m4_stage1.png`).

## Script map

- `scripts/run_forward.py` — M0 rollout (`--solver semi_implicit|vbd`)
- `scripts/check_grad.py` — M1 FD-vs-tape check (exit 1 + CEM advice on failure)
- `scripts/recover_synthetic.py` — M2 single-param (`--method grad|cem`)
- `scripts/recover_full.py` — M3 8-dim (`--method grad|cem|lm --fix ... --starts N --init-from ...`)
- `scripts/probe_identifiability.py` — FD Hessian + 2D slice (use GN variant in
  `outputs/identifiability_gn.npz` for conclusions)
- `scripts/run_m4_pipeline.py` — M4 stage 1 end-to-end

## M4 stage 2 groundwork (2026-07-22)

- **i2v model choice** (searched current landscape): bake-off of three candidates on
  OUR metric (render I0 from known theta_true -> generate -> track -> recover;
  score by theta error). External physics benchmarks are unreliable (Physics-IQ
  Verified audit found ~58 % contaminated samples). Candidates + rationale:
  - `Wan-AI/Wan2.2-TI2V-5B-Diffusers` — first-frame faithfulness (our anchor), Apache 2.0,
    fast plumbing model; `WanImageToVideoPipeline` in diffusers 0.39.
  - `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` — reputed best
    physical/cloth motion; `HunyuanVideo15ImageToVideoPipeline`.
  - `nvidia/Cosmos3-Nano` (16B omnimodel, June 2026, OpenMDW) — physics-focused;
    serves via vLLM-Omni server, client wrapper stubbed pending bake-off.
  - A6000 = sm_86 Ampere: **no FP8** — bf16 paths only. 480p output is enough
    (we only extract tracks). Static-camera prompt suffix + anti-camera-motion
    negative prompt in `src/video/i2v.py`.
- **conda env `video`** (torch 2.13 cu126, diffusers 0.39) — separate from `warp`.
  Weights at `HF_HOME=/home/nas5/jooyeolyun/hf_cache` (NAS; local disk too small).
  NAS I/O saturates during big downloads — expect slow git/du meanwhile.
- **Real assets** (`scripts/fetch_assets.sh`, binaries gitignored): GSO scans
  (CC-BY 4.0) — 6 rigid (lion toy 11.5 cm, teapot, clamp, file sorter, dino,
  shark) + 2 cloth-like (Provence bath towel 36 cm, braided cushion); PolyHaven
  CC0 wooden table + studio HDRI. All load via trimesh at true metric scale.
  `scripts/inspect_assets.py` drop-tests in Newton (XPBD + CollisionPipeline):
  lion settles at z=0.021 — PASSED. Newton 1.4.0 collision API:
  `pipeline = CollisionPipeline(model); contacts = pipeline.contacts();
  pipeline.collide(state, contacts)` per step; shape density via
  `ModelBuilder.ShapeConfig(density=...)` passed as `cfg=`.
- GSO towel/cushion are *scans of resting cloth* — usable for appearance/geometry
  reference; cloth sim needs retopology onto a regular grid (or use as rigid).

## M4 stage 2 — Wan plumbing test (2026-07-22) PASSED

`scripts/run_i2v.py --backend wan5b --image outputs/m4_I0.png --seeds 0 1 2`
(49 frames, 480x480, 35 steps, GPU 0):

- **First-frame faithfulness: |f0 − I0| = 1.1/255** — Wan reproduces the conditioning
  frame almost exactly; barycentric track attachment on I0 remains valid.
- **Camera static** across all seeds (prompt suffix + negative prompt worked).
- **Motion plausible and diverse**: seeds 0/1 gentle waving, seed 2 violent flapping
  with fold shadows. Texture pattern stays coherent.
- **LK tracks generated video**: survival 96 % (seed 0) / 65 % / 25 % (seed 2,
  violent motion + appearance drift — still 75 tracks, above the >=20 floor).
  Confirms CoTracker upgrade is about the harsh-motion tail, not a blocker.
- Cost: ~60 s per 49-frame video after a one-time 13.5-min weight load (NAS was
  saturated by concurrent downloads; loads are fast once NAS is idle).
- Note: our matplotlib renderer produces a WHITE background (`ax.set_facecolor`
  is ignored with `axis('off')`) — the prompt said "black background" and Wan
  followed the image, not the text. Harmless here; align prompt text with the
  actual render for the bake-off.
- Weights ready: Wan2.2-TI2V-5B (33 GB) and HunyuanVideo-1.5-480p_i2v complete
  in HF_HOME; Cosmos3-Nano downloading.

## Bake-off (2026-07-22, `scripts/run_bakeoff.py`)

Metric correction vs the original plan: the i2v model only sees a static I0 and
CANNOT know theta_true — so we score by **distance to the physical manifold**
(post-fit 2D RMS after multi-start LM projection onto our Newton family, sim
retimed to 24 fps / 48 frames / substeps 96 = dt 4.3e-4), plus trackability and
motion magnitude (near-static videos fit trivially -> flagged degenerate below
20 px median motion). Recovered theta is descriptive: it reveals what physics
the video model *implied*.

Wan2.2-TI2V-5B leg (3 seeds):
- seed 0: 278 tracks (96 %), motion 138 px, fit RMS **14.7 px**
- seed 1: 189 tracks (65 %), motion 83 px, fit RMS **12.4 px**
- seed 2: 71 tracks (25 %), motion 32 px, fit RMS **1.4 px** (calm subset —
  violent-motion tracks self-reject via the fb gate; step-8 rejection working)
- Reference: real-sim video fits at 4 px (LK noise floor). Wan's active motion
  sits ~3x above the manifold.
- **Implied physics = video-model bias, quantified**: implied gravity
  -0.6 .. -1.8 m/s^2 across ALL seeds (5-15x weaker than Earth — "dreamy
  slow-mo cloth"), with inflated wind (24-37 m/s) and stiffness on active
  seeds. Projection onto the manifold doubles as a physics audit of the
  generative model. (Caveat: 2 s horizon, fixed 1 Hz forcing, single view.)

Ops notes: vllm-omni pip package exists but needs core `vllm` separately and
~40 GB free VRAM for Cosmos3-Nano (16B bf16) — blocked on shared-box memory,
weights + HTTP client ready. vllm-omni also downgrades diffusers 0.39->0.38
(both our pipelines still present). HunyuanVideo15ImageToVideoPipeline takes
NO height/width/guidance kwargs (resolution from input image); 49f@480p peaks
~37 GB -> needs vae tiling + expandable_segments on a shared GPU.

HunyuanVideo-1.5 legs (2 prompt variants, 3 seeds each):
- **Scene hallucination**: with the original prompt ("...pinned to a pole"), Hunyuan
  materializes a physical flagpole + tripod stand from frame ~12 and reframes the
  flag (Wan ignored the same phrase). Mean fit RMS 19.8 px (seed 1: 40 px).
- Scene-neutral prompt removes the pole but Hunyuan STILL re-stages (cloth drifts
  to center, rescales, faint clothesline). Tracks survive (79-95 % — LK happily
  follows rigid drift) but a pinned-edge Newton flag cannot translate -> mean fit
  RMS **27.0 px** (seed 11: 56 px). Scene drift converts to fit error, not track
  death: the metric penalizes exactly what breaks the pipeline.
- Hunyuan's local wrinkle dynamics LOOK more cloth-like than Wan's, but it treats
  I0 as a suggestion, not a locked camera view. 10x slower than Wan (600 s vs
  60 s per 49-frame video).

FINAL (Cosmos3 not scored — serving blocked on VRAM/core-vllm):

| backend  | prompt   | mean fit RMS | mean survival | mean motion |
|----------|----------|--------------|---------------|-------------|
| wan5b    | default  | **9.46 px**  | 62 %          | 84 px       |
| hunyuan  | pole     | 19.76 px     | 55 %          | 53 px       |
| hunyuan  | neutral  | 27.04 px     | 88 %          | 35 px       |

**Verdict: Wan2.2-TI2V-5B is the production i2v backend for M4 stage 2** —
closest to the physical manifold, zero scene edits, static camera, most motion,
10x cheapest. Hunyuan disqualified by scene re-staging under both prompts.
Cosmos3 to be scored if/when a ~40 GB GPU frees up.

Then: real-asset variant (towel on table) with Wan as the backend.

## Cosmos3 serving: BLOCKED on GPU driver (2026-07-22, investigated to root cause)

Dependency chain, each link verified against actual wheels:
- Cosmos3 model code exists only in `vllm-omni >= 0.22` (0.22 released 2026-06-08,
  after Cosmos3's 05-31 launch; confirmed `vllm_omni/diffusion/models/cosmos3/` in
  the 0.24 wheel). vllm-omni declares NO vllm pin in metadata (runtime coupling).
- Every `vllm` PyPI wheel from **0.20.0 onward links libcudart.so.13** (CUDA 13);
  the last CUDA-12 build is **0.19.1** (March, pre-Cosmos3). Checked by readelf on
  the actual .so files. wheels.vllm.ai now hosts cu130 only (no cu128 variants).
- This box: driver 575.57.08 = CUDA 12.9. torch cu130 refuses to initialize
  ("driver too old"). torch 2.11+cu128 works fine.

Unblock paths: (a) **driver upgrade to >= 580 (CUDA 13)** — admin action on the
shared box; or (b) build vllm >= 0.22 from source against cu128 torch in the
`cosmos` env (torch 2.11.0+cu128 already installed there and CUDA-verified) —
multi-hour compile, uncertain. Weights (33 GB) + HTTP client remain ready.

**Incident + fix**: `pip install vllm` into the `video` env silently replaced
torch 2.13+cu126 with 2.11+cu130 (unusable on this driver) and broke the
diffusers stack mid-flight (also cost us the GPU 0 claim — another user grabbed
it during the breakage window). Restored: vllm/vllm-omni uninstalled from
`video`, torch 2.13+cu126 + torchaudio reinstalled, Wan/Hunyuan pipelines
verified. RULE: **never install vllm into the diffusers env** — it force-swaps
torch. Serving stacks get their own env (`cosmos`).

## GPU holds (2026-07-22)

`scripts/run_gpu.py` claims freed GPUs (~42 GiB each) and idles until
`touch /tmp/release_gpu<N>`. Currently holding GPUs 1-4. A Monitor watches
nvidia-smi for newly freed GPUs (<1.5 GB used). GPU 0 freed but was lost to
another user's job within the env-breakage window.

## Open items / next

- M4 stage 2: real/generated video — needs an i2v model choice (step 3) + initial-frame
  render of the real asset; tracker upgrade to CoTracker if LK is insufficient on real
  footage.
- Occlusion in the renderer is painter-sorted only; folds may confuse LK on harsher
  motions — the fb gate currently absorbs this by killing tracks.
- eval/ metrics harness still a stub; formalize recovery-error metrics across seeds.

## M5 — real-asset rigid-body inverse (2026-07-22)

First inverse recovery on a **real scanned object** (GSO teapot, decimated to
1500 faces). Drops from 0.35 m with an initial launch velocity + spin onto a
ground plane, falls / bounces (restitution) / slides (friction). We recover the
object's physical attributes from its pose trajectory. This is the "real asset
data" (`[scale]`) milestone, single-object.

theta = [log_density, mu, restitution, v0(3), w0(3)] (9). Contact material and
mass/inertia are mutated in place on the finalized model (no rebuild per eval):
`shape_material_mu`, `shape_material_restitution`, `body_mass`/`body_inv_mass`/
`body_inertia`/`body_inv_inertia` (mass & inertia scale linearly with density).

Newton 1.4.0 rigid API facts (verified):
- `body_q` = transform (pos xyz + quat xyzw); `body_qd` = spatial velocity,
  **(linear v[0:3], angular w[3:6])** — NOT the screw-theory (w, v) order.
  Verified empirically (`qd=[1,0,0,0,0,0]` translates in x; `[0,0,0,1,0,0]`
  rotates about x). Getting this wrong made the object launch upward.
- `SolverXPBD(model, iterations, enable_restitution=True)`; contact via
  `CollisionPipeline`; ground friction via `add_ground_plane(cfg=ShapeConfig(mu=))`.
- Observation = 8 bbox-corner marker world positions per frame (translation +
  orientation), a rigid analog of the cloth vertex trajectory.

**M5.1 gradient check** (`scripts/check_grad_rigid.py`): XPBD tape gradient
through contact is **exactly zero** (FD is nonzero) — same as VBD cloth. Recovery
uses the solver-agnostic FD-Jacobian LM. (Validates the M1 architectural bet:
building LM/CEM paid off again.)

**Density is a (near-)gauge freedom — the rigid analog of the cloth scale gauge.**
In ideal rigid contact against a fixed plane, both free-fall (a=g) and the
collision laws (restitution, Coulomb friction as velocity/impulse ratios) are
mass-independent, so the trajectory should be density-invariant. Empirically,
x4 density perturbs markers by only ~9 mm over a ~0.2 m trajectory — weakly
observable, and that residual observability is a *numerical artifact* of XPBD's
compliant contact (fixed contact stiffness not scaled with mass), not real
physics. Prediction: density is the least-identifiable parameter; friction and
restitution are observable only through their contact events (sliding / bounce);
initial velocities are strongly observable from the early free-flight arc.

### M5 results

7 multi-start LM runs (parallel across GPUs 1-4, ~20-30 min each). **Best fit
RMS 17.7 mm** — not tight (~10 % of the ~0.15 m trajectory). The story is in
*which* parameters recover, and it matches the identifiability probe exactly.

Recovered vs true (Gauss-Newton per-param marker sensitivity, m per unit theta):

| param        | true | recovered | sensitivity | verdict |
|--------------|------|-----------|-------------|---------|
| v0y          | 0.15 | 0.17      | 0.42        | good (observable) |
| v0x          | 0.60 | 0.70      | 0.27        | good |
| w0z          | 1.50 | 1.15      | 0.14        | fair |
| restitution  | 0.55 | 0.27      | 0.11        | poor |
| v0z          | 0.00 | 0.06      | 0.11        | ok |
| mu           | 0.40 | 1.49      | 0.074       | poor |
| w0y          | 3.00 | 1.44      | 0.044       | poor |
| density      | 800  | ~0 (floor)| 0.0084      | UNRECOVERABLE (gauge) |

cond number 4.7e4. **Recovery quality tracks observability almost monotonically**:
the free-flight launch velocities (highest sensitivity) recover to ~15 %; the
contact-dependent params (restitution, mu) and the roll-axis spin w0y are poor;
**density is driven to the clamp floor** (sensitivity 50x below v0y) — direct
confirmation of the density gauge freedom. The recovered COM tracks the true
free-flight arc and diverges only after the first bounce (see
`outputs/rigid_recover.png`), i.e. residual concentrates exactly in the
contact-governed, weakly-observable phase.

**Honest limitation**: the 17.7 mm floor (even with 7 starts) is the rugged
rigid-contact landscape — bounce timing is chaotic in the parameters, so LM traps
in local minima; the observable params aren't nailed to sub-mm the way exact-3D
cloth recovery was (<=0.07 %). Levers: longer / multi-view / multi-drop
observation to raise contact-param sensitivity, contact-event-aware loss
weighting, and more starts. The scientific point stands regardless: **you recover
what the motion observes, and the identifiability probe says in advance what that
is** — here, launch kinematics yes, material/density no.


Perf: rollout 8.7 s (72 frames x 24 substeps, XPBD 15 iters). LM ~160 s/iter
(18 rollouts/Jacobian). **Multi-start run parallelized across GPUs 1-4** (one
start per GPU) via `scripts/recover_rigid.py --start K` + `--aggregate` — the
held GPUs put to productive use. `scripts/run_gpu.py` (was gpu_hold.py) claims/
holds idle GPUs; release with `touch /tmp/release_gpu<N>`.

## M6 — real MULTI-asset scene (2026-07-23)

Two real scanned objects (GSO teapot + great-white-shark) launched toward each
other on a real wooden table, colliding and settling. This is the "Scale"
milestone: multi-object + real asset. theta = 6/object x 2 = 12
[log_density, mu, restitution, v0(3)] each.

Engineering (Newton 1.4.0):
- **Object-object contact needs SDF.** Triangle-mesh vs triangle-mesh contact
  between two scanned meshes explodes (non-finite by frame 7). `mesh.build_sdf(
  max_resolution=64)` on each object -> robust volumetric contact, stable.
- **Objects fall through a static triangle-mesh table.** Object-vs-static-mesh
  contact doesn't catch them. Fix: a **static box collision proxy** for the
  tabletop (exact, stable) + the real table mesh kept only for rendering — a
  standard visual-mesh / collision-proxy split (`render_scene.py`).
- Kinematic tabletop is body 0; object bodies are offset -> index by obj_body[].
- Perf: 7.5 s/rollout (60 frames x 24 substeps, SDF res 64, XPBD 16).

### M6 results

4 parallel multi-start LM (one stuck start killed; best-of-3). **Best fit
13.9 mm RMS — better than single-object M5 (17.7 mm)**; recovered paths track the
true collision trajectories (`outputs/scene_recover.png`).

Recovery again tracks observability (GN sensitivity in parens, m/unit theta):
- Launch velocities recovered well: v0x within ~15-18 % (teapot 1.70->1.46,
  shark -1.50->-1.23), v0y/v0z ~0 correct. (sens 0.16-0.43)
- Contact params poor: mu, restitution off ~2x (sens ~0.23).
- **Densities are the two least-observable params** (sens 0.023 / 0.059): teapot
  900->1588 (76 % off), shark 400-> ran to the 1e5 clamp ceiling (unidentified).

**Does object-object collision break the density gauge?** Partially. The
density-swap forward test (swap 900<->400) moves markers 36 mm, so the collision
*does* couple the masses. But locally the coupling is weak: the density
RATIO direction (d0 up / d1 down) has only **1.4x** the loss curvature of the
overall SCALE direction (both up) — both are ~20x below the velocity
sensitivities. So the collision *softens* the per-object density gauge rather
than cleanly breaking it; density stays the weakest-observed quantity and the
light shark's density runs to the clamp. To strongly identify relative density
you'd need a collision-dominated scene (repeated/harder impacts, collision as the
primary motion) rather than launch-and-settle where kinematics dominate.

Takeaway (consistent M5->M6): **you recover what the motion observes.** Launch
kinematics: yes. Contact material + density: weak, and weaker still for the
lighter object. The identifiability probe predicts the ranking every time.

### GPU credit / efficiency

`davian credit`: A6000 bills **9 credit/hr PER GPU regardless of utilization**.
Multi-start LM starts are launch-bound (~14 % GPU, ~400 MB) -> spreading N starts
across N GPUs costs Nx for no speedup. Pack them onto ONE GPU. Don't hold idle
GPUs. (Corrected the earlier hold-everything approach.)

## M7 — collision-dominated gauge + real backgrounds (2026-07-23)

### Real backgrounds (city scene)

Replaced the white-background renders with a **photorealistic city scene**
(`scripts/blender_city.py`, Blender 4.2 + Cycles GPU, ~2 s/frame): PolyHaven CC0
city HDRI (pretville_street) for sky/lighting/backdrop + painted wooden bench +
Victorian street lamp + fire hydrant + a flagpole flying **our cloth-sim flag**
(exported to OBJ by `export_flag_obj.py`). `outputs/city_scene.png`. This is the
realistic I0 the M4 i2v pipeline wanted — a flag in an urban context, not on
white. Assets fetched reproducibly by `fetch_assets.sh` (gitignored binaries).
Gotcha: Blender's OBJ importer defaults to Y-up; pass `up_axis="Z"` to keep our
Z-up sim mesh upright (else the flag lies flat).

### Collision-dominated scene: does it break the density gauge?

`configs/collide.yaml`: two similar-mass scanned objects (shark + triceratops)
head-on on a low-friction tabletop, near-elastic (e=0.7). Post-collision velocity
split depends on the mass ratio. `scripts/probe_density_gauge.py` measures the
density scale-vs-ratio curvature.

- **Observability: gauge BROKEN.** Density sensitivity jumps from ~0.03 (M6
  launch scene) to 0.08-0.60 (comparable to velocity sensitivity). The GN
  density RATIO direction is **3.4x** the SCALE direction (M6 was 1.4x) -> the
  collision makes relative density observable. Density-swap moves markers 1.4 m
  (M6: 36 mm).
- **But recoverability: LM FAILS.** `recover_scene.py --config collide.yaml`
  (4 starts): every recovered param = its init value; best RMS 224 mm (M6:
  13.9 mm); cond 1.2e7. LM trace shows lambda -> 3e5 -> 3e8 at iter 0 with
  |step| ~ 4e-6: **no acceptable descent step exists** at the start. The violent
  collision makes bounce timing chaotic in the parameters, so the FD Jacobian is
  noise. Observability and optimizability are in direct conflict here.

**The observability-optimizability tradeoff (bracketed):**
- M6 gentle collision: smooth landscape, LM recovers to 13.9 mm, but density
  nearly unobservable (ratio 1.4x scale) -> density not recovered.
- M7 violent collision: density observable (ratio 3.4x scale) but chaotic
  landscape -> LM makes zero progress.
Making contact informative enough to reveal density also destroys the smoothness
local optimization needs. Resolution attempt: a GLOBAL gradient-free method
(CEM) that tolerates chaos -> `recover_scene_cem.py`.

### CEM verdict: the gauge is observable but UNRECOVERABLE (by any method we have)

CEM (global, gradient-free, pop 28 x 16 iters, 4496 s) ALSO fails on the
collision scene:
- best fit RMS **1130 mm** (worse than LM's 224 mm — CEM wandered further from
  init); loss stuck at ~32 across all iterations, no descent.
- density RATIO obj0/obj1: true 2.0, **recovered 0.52** — missed and inverted.
- velocities wrong, some sign-flipped.

**Definitive result: observability != recoverability.** The collision-dominated
scene makes density observable (ratio 3.4x scale, swap moves markers 1.4 m) yet
recoverable by NEITHER local (LM: frozen at init) NOR global (CEM: stuck in the
chaotic loss sea) optimization. The very violence that reveals density also makes
the loss landscape a needle-in-fractal-haystack: the true optimum is a
measure-zero basin surrounded by deceptive ~32-loss minima, and high sensitivity
(the 1.4 m swap) is the *symptom* of that intractability, not a promise of easy
recovery.

The bracket is now closed:
- M6 gentle collision: smooth landscape, LM recovers to 13.9 mm, density
  unobservable (ratio 1.4x) -> not recovered.
- M7 violent collision: density observable (ratio 3.4x) but landscape chaotic ->
  LM and CEM both fail.
There is no free lunch: for this rigid-contact system, the regime that makes a
material parameter observable is the same regime that destroys the smoothness
optimization needs. Recovering density would need a different lever entirely —
higher-temporal-resolution observation of the impact, a smoother/analytic contact
model, or a physics-informed parameterization that fits the momentum-conservation
relation directly rather than the full chaotic trajectory. This connects to the
material-vs-forcing distinction: density only modulates motion through the
collision, and that modulation is non-smooth, so it resists trajectory-fitting.

## M7b — high-time-resolution observation: the root cause surfaces (2026-07-23)

Tried the proposed lever — observe the collision at high frame rate to recover
density. `configs/collide_hires.yaml` (obs_stride=1, ~1500 Hz), two comparable-mass
objects colliding MID-AIR (ground env, no table) so only the mutual impulse acts.

1. **Dense positions + trajectory fit still fails.** LM on the densely-sampled
   position residual freezes at iter 0 (lambda->1e7) exactly like the 60 Hz case.
   Densely sampling a chaotic trajectory is still chaotic — resolution doesn't
   smooth the landscape.
2. **The right use of high-res is to MEASURE the impulse, not fit the trajectory.**
   The velocity change at contact is a smooth algebraic function of the mass ratio:
   m0*dv0 = -m1*dv1. High-res resolves the impact cleanly (`recover_density_
   momentum.py`; velocities in `outputs/momentum_nonconservation.png` left panel).
3. **But the momentum law it relies on is VIOLATED by the solver.** Using the true
   asymptotic velocities (start = exactly the launch v0 = free flight; end = fully
   separated), total horizontal momentum drops **57 %** (0.162 -> 0.070 kg·m/s)
   through the collision — see the flat-then-collapse curve in the figure's right
   panel. Real collisions conserve momentum exactly (restitution loses energy, not
   momentum). The loss is stable across xpbd_iterations 18/50/100 (~55-57 %), so it
   is fundamental to XPBD's contact model, not a convergence artifact. The measured
   mass ratio comes out 1.26 vs true 2.48 — off by ~2x, exactly the momentum leak.

**Root cause of ALL contact-recovery failures, unified.** Newton's position-based
XPBD contact is built for fast, stable FORWARD simulation, not for physical
fidelity of the contact impulse. Across the milestones its contact is:
- **non-differentiable** (tape gradient exactly 0 — M5),
- **non-scale-invariant** (absolute density leaks into the collision — M7), and
- **non-momentum-conserving** (57 % horizontal-momentum loss — M7b).
Any inverse method that relies on the contact impulse being physically correct —
gradient descent, or momentum-conservation measurement — fails through this solver.
It is not chaos alone and not our estimator; the discrete contact is unfaithful.

**The real fix (recommended next step):** a momentum-conserving contact solver.
Newton ships `SolverMuJoCo` (MuJoCo's contact is physically grounded), but it needs
`pip install mujoco` + `mujoco_warp` and the scene re-expressed for that backend —
a separate validation. Alternatively an impulse/LCP rigid solver. With momentum
conserved, the high-res momentum-measurement recovery of the density ratio should
work directly (it is a linear read-off, no chaotic optimization). The high-res
OBSERVATION is correct and sufficient; the SOLVER is what currently defeats it.

## M8 — differentiable penalty contact in Warp: the fix (2026-07-23)

Root cause (M7b) was Newton's XPBD contact, not Warp. Fix: write contact
ourselves as a custom Warp kernel — the same pattern as our wind force — so it is
differentiable AND momentum-conserving by construction. No Warp patch needed;
Warp's autodiff handles the custom kernel directly. `src/sim/diff_collide.py`,
`scripts/diff_collide_recover.py`. Spheres (analytic penetration), mass =
density * (4/3 pi r^3), semi-implicit integration, penalty contact
F = k*overlap - c*vrel along the normal.

Two guarantees BY CONSTRUCTION:
- **momentum conservation**: the pairwise force is antisymmetric (F_ij = -F_ji),
  so sum of contact forces = 0. Measured drift over the whole rollout:
  **1.2e-6** (machine precision) vs XPBD's **57 %** loss.
- **differentiability**: Warp auto-generates the kernel adjoint, so the trajectory-
  loss gradient flows back to density. Tape grad matches FD to **3e-4 rel err**
  vs XPBD's **exactly 0**.

Result — two spheres (equal radius, densities 800/400, mass ratio 2) collide
head-on; recover sphere-0's density from the trajectory by GRADIENT DESCENT
(Adam, 60 iters): **812 vs true 800 (1.5 %)**, ratio 2.03 vs 2.00. Loss
1.2e-2 -> 1.4e-5. This is the density recovery from a collision that XPBD
(LM frozen at init) and CEM (ratio 0.52, inverted) could not do. Figure:
`outputs/diff_collide.png` (flat momentum) vs `outputs/momentum_nonconservation.png`
(XPBD 57 % leak).

The trade the user accepted: stiff penalty (k=6000) needs small dt (5e-4 s),
so it is slower than XPBD — but exact, momentum-conserving, and differentiable.
Absolute mass scale remains a true gauge (elastic collision is scale-invariant);
the collision makes the RATIO observable, which is exactly what we recover.

**Next (extend to real assets):** swap the sphere-overlap term for an SDF
penetration query. Newton meshes already expose `mesh.build_sdf()` (used for the
M6 forward contact); querying that SDF in a Warp kernel gives differentiable
penetration for real scanned geometry, keeping momentum conservation and
gradients. The proof of concept is done; mesh-SDF differentiable contact is the
remaining engineering. This is the through-line resolved: with a differentiable,
momentum-conserving contact written in Warp, contact parameters become
recoverable — the simulator, not the observation or the optimizer, was the wall.

## M9 — differentiable contact on REAL mesh assets (2026-07-23)

Extended M8's differentiable contact from spheres to real scanned geometry.
`src/sim/diff_collide_mesh.py`, `scripts/diff_collide_mesh_recover.py`.

Approach chosen (after a false start): **sphere-covering**. Warp's mesh_query
(`mesh_query_point_sign_winding_number`) conserves momentum but its adjoint is
BROKEN — the discrete BVH face-selection is non-differentiable, so tape grad
(7.6e-3) disagreed with FD (-0.24, wrong sign). The correct-and-robust fix:
approximate each real mesh by a covering of interior spheres (trimesh
`voxelized(pitch)`, surface shell — `.fill()` needs scipy which errored), then
use the analytic sphere-sphere penalty contact from M8 between the two sphere
sets. Every contact term is then a smooth function of position -> gradient flows;
equal-and-opposite pair forces -> momentum conserved. The sphere set traces the
real shape (124 spheres/object for a triceratops at pitch 0.012).

Two bugs cost a cycle (both broke the gradient, not the forward):
- **force array not zeroed** before the per-step atomic_add accumulation -> forces
  accumulated across rollouts in the Adam loop. Fix: `force[t].zero_()` each step.
- **in-place gravity** (`force[i] = force[i] + g*mass`) broke Warp's tape adjoint.
  Fix: fold gravity into the integrator as an acceleration (`v += (f*inv_m + g)*dt`),
  no in-place write. (This is the NVIDIA-blog rule: don't overwrite arrays in place
  on the tape.)

Result (two triceratops scans, densities 800/400, head-on):
- momentum drift over rollout **7.45e-9** (machine precision)
- tape grad matches FD to **6e-4** rel err (GRADIENT FLOWS through real-mesh contact)
- density recovered by gradient descent: **775 vs true 800 (3.1 %)**, ratio 1.94 vs 2.0
Figure `outputs/diff_collide_mesh.png` (collision + recovery curve).

So the whole arc closes: differentiable + momentum-conserving contact, now on REAL
scanned assets, recovers density from a collision — the thing XPBD/CEM/momentum-
measurement could not. Sphere-covering is the pragmatic differentiable contact;
the "exact" alternative is sampling a precomputed SDF volume with
`wp.volume_sample_grad_f` (differentiable) — Newton's `mesh.build_sdf().sparse_volume`
provides it; deferred as it needs sparse-volume index-transform plumbing, and the
sphere covering already delivers differentiable real-mesh contact.

Remaining for full realism: 6-DOF (orientation + inertia + torque; contact points
already give torque arms) and friction. Translation-only suffices for the density-
from-collision result because linear-momentum exchange is what carries the mass ratio.

## M10 — 6-DOF rotation + friction on the mesh contact (2026-07-23)

`src/sim/diff_collide_6dof.py`, `scripts/diff_collide_6dof_recover.py`. Full rigid
dynamics, still differentiable + momentum-conserving, on real (sphere-covered) meshes.

Added over M9 (translation-only):
- **orientation** (quaternion) + **angular velocity**, integrated with Euler's
  equation: world inertia I_w = R (mass*G) R^T (G = per-unit-mass inertia of the
  sphere cloud, precomputed), gyroscopic term omega x (I_w omega) included,
  quaternion update q += 0.5*omega_quat*q*dt, renormalized.
- **torque**: each contact force applied at the shared contact point (midpoint of
  the overlapping spheres) with arm r x f -> spins the bodies.
- **friction**: regularized Coulomb f_t = -mu*f_n*v_t/(|v_t|+1e-3), smooth ->
  differentiable, opposes tangential slip; v_t uses the contact-point velocity
  v_com + omega x r, so it couples rotation.
- momentum conserved for BOTH linear and angular because the force pair (f, -f)
  acts at the same contact point.

Warp API used: wp.quat_rotate, wp.quat_to_matrix, wp.transpose, mat33*mat33,
mat33*vec3, quat*quat, quat*scalar, quat+quat, wp.normalize(quat) — all compiled
first try; the module ran without API friction.

Result (two triceratops scans, y-offset for an off-center spin-inducing hit,
mu=0.5, densities 800/400):
- linear momentum drift **2.2e-8**, angular momentum drift **2.1e-7** (both conserved)
- rotation induced: obj0 **91 deg**, obj1 **83 deg** (torque + friction active)
- tape grad matches FD to **7e-4** rel err (gradient flows through the full 6-DOF+friction rollout)
- density recovered **793.8 vs true 800 (0.8 %)**, ratio 1.98 vs 2.0 — BETTER than
  translation-only M9 (3.1 %): rotation/friction coupling adds observability.
Figure `outputs/diff_collide_6dof.png`.

The physics engine is now: 6-DOF rigid bodies, real scanned geometry, penalty
contact with Coulomb friction, fully differentiable in Warp, exactly conserving
linear + angular momentum — and it recovers material parameters from a collision
by gradient descent. Every gap from the XPBD root-cause (M7b) is closed:
differentiable, scale-consistent (ratio recovered), momentum-conserving.
Remaining niceties: friction/restitution as recoverable theta (the machinery is
there — they enter the differentiable force), multi-body (N>2) contact broad-phase,
and coupling this contact into the cloth/scene pipeline for full assetization.

## M11 — friction + restitution as recoverable parameters, and videos (2026-07-23)

Made mu (friction) and cd (normal damping = restitution) differentiable wp.arrays
in `diff_collide_6dof.py` (kernel reads mu_arr[0], cd_arr[0]); recover all three
contact params jointly: `scripts/diff_collide_params_recover.py`.
theta = [log_density_0, mu, log_cd], density_1 fixed (mass-scale gauge).

Key fix — **fit orientation, not just position**. First attempt recovered density
(0.1%) and cd but mu wandered (0.2-1.8, loss insensitive to it): the position-only
loss barely sees friction, because friction mainly changes ROTATION. Added an
orientation term `accum_quat_loss` (1 - (q.q_target)^2, weight 0.02) -> friction
became observable. Also: near-head-on hit (offset +/-0.015) to excite restitution,
pre-spin obj0 (14 rad/s) to excite friction, lr 0.04 + keep-best for stability.

Result (true d0=800, mu=0.55, cd=12; init 1600/0.15/40):
- density   **800.0 (0.0%)**
- friction  **0.535 vs 0.55 (2.8%)**
- restitution(cd) **12.02 vs 12.0 (0.2%)**
All three contact parameters recovered from ONE collision by gradient descent.
(Effective-e MEASUREMENT is unreliable for glancing mesh collisions -> report the
cd parameter directly, which is what's identified.) Figure params_recover.png.

**Videos**: no imageio/ffmpeg in the warp env, but matplotlib PillowWriter makes
animated GIFs (embed inline in the artifact via base64). `render_collision_video.py`
renders the two real triceratops meshes tumbling through the 6-DOF collision
(projected, shaded, painter-sorted) -> `outputs/collision.gif`. img_tag() now
sets image/gif mime so the GIF animates in report.html.

Identifiability note: density<-mass ratio (momentum), friction<-spin transfer
(needs orientation in the loss + tangential slip), restitution<-normal bounce
(needs a head-on component + inelastic regime). The off-center-hit-with-pre-spin
scene excites all three; a purely head-on or purely glancing hit would leave one
weakly observed.

## M12 — full video pipeline end-to-end on real assets (2026-07-23)

`scripts/full_pipeline.py`. Every piece wired together, on real scanned meshes:
1. real GSO meshes -> 6-DOF+friction collision (differentiable contact), theta_true
2. RENDER to a textured RGB video (per-face grayscale so LK has features)
3. LK point-TRACK the video (129/153 points survive all 51 frames)
4. ATTACH each frame-0 track to the nearest object vertex -> (body index, body-frame point)
5. RECOVER density by differentiable simulation + differentiable camera projection
   (`project_rigid_loss` kernel: world = pos_b + quat_rotate(rot_b, pt_local); project;
   2D-track residual) — gradient flows through projection AND the 6-DOF contact rollout
6. overlay GIF: LK tracks (red) vs recovered-sim reprojection (cyan) on the video

Result: density_0 recovered **838 vs true 800 (4.8 %)** from the VIDEO — vs ~0-3 %
from direct 3-D. The extra error is the LK tracking-noise floor (the same
observation-quality limit found in M4), not the optimizer or the physics. Recovery
loss was a bit noisy (2-D tracks) but converged. `outputs/full_pipeline.gif` shows
the recovered simulation reprojecting onto the tracked points.

This closes the loop the project set out to build: a real 3-D asset, its motion
observed only through a rendered/tracked VIDEO, and its physical parameters
recovered by a differentiable Warp simulation projected back to the image — with a
differentiable, momentum-conserving contact (M8-M11) as the engine that made
contact-rich recovery possible at all. Steps 2,4,6,7,8,9 of the architecture,
running on real assets, verifiable, with video output.

Wild extension (not run here): swap step 2's render for a Wan i2v generation
(I0 -> V*) — the plumbing (i2v backends, tracker, projection, recovery) is all in
place; the only change is the video source.

## M13 — realistic staged scene, physics from a photorealistic video (2026-07-23)

The user's ask: real assets in a real *scene*, not floating in a void. Built the
drop pipeline: `prep_drop_scene.py` (warp) -> `blender_drop.py` (Cycles) ->
`recover_drop_scene.py` (warp). A real scanned triceratops falls onto the real
wooden table in the city street (HDRI + lamp + hydrant props), rendered
photorealistically; physics recovered from that video.

- `src/sim/diff_drop.py`: single 6-DOF body + differentiable ground-plane
  penalty+friction contact (restitution=normal damping cd, friction=mu, both
  recoverable). FD-verified gradient (tape -0.0190 vs FD -0.0190).
- **Camera match exact**: Blender camera (sensor_fit=HORIZONTAL, angle=fov) ==
  src/render/camera.py pinhole; projected mesh lands on the rendered object.
- RESULT: **restitution recovered cd 7.43 vs true 7.0 (6%)** from the photorealistic
  bounce video. Friction weakly observed (79% off) — the object barely slides, and
  adding slide speed loses the LK tracker (friction-vs-trackability tension).
- density is a gauge under gravity (M5), so not recovered here.

Hard-won lessons (all cost cycles):
1. **NaN from invalid tracks**: LK fills lost tracks with NaN; NaN*valid(0)=NaN in
   float poisons the loss. Fix: np.nan_to_num the tracks (valid mask does the rest).
2. **Static-background tracks**: the frame-0 on-object filter also kept static
   background points behind the object; they gave 147px error at TRUE params. Fix:
   keep only tracks whose screen motion > 0.6x the object's (they ride the object).
3. **Contact-gradient explosion**: backprop through EXTENDED/continuous contact
   (a settling object) NaNs. Fix: stage a single clean BOUNCE (object airborne,
   one brief contact) and observe drop->bounce->rise; clamp damping to the bouncy
   regime so the optimizer never enters the settling regime.
4. **Fast fall vs LK**: kept motion <~13 px/frame (within LK range) by tuning drop
   height + frame rate; faster motion loses tracks (the CoTracker case from M4).

This is the milestone the user wanted: a believable staged scene (real object,
real table, real street), a photorealistic video, and a real physical parameter
(bounciness) recovered from it by the differentiable Warp pipeline.

## M14 — multi-video test: different bounciness recovered per video (2026-07-24)

Parameterized the drop pipeline by env vars (SCENE=subfolder, CD=true restitution
damping) across prep/blender/recover, shorter gradient-stable horizon (NSTEPS=1900,
STRIDE=26 — ends soon after impact so backprop avoids the continuous-contact
explosion; verified finite gradient for cd 4->28). Ran three scenes, same object/
drop, different material:
- Bouncy  (true cd=4):  recovered 3.38  (15%)
- Medium  (true cd=9):  recovered 7.61  (15%)
- Dead thud (true cd=18): recovered 12.96 (28%)
Recovered bounciness tracks truth monotonically; the dead case is fuzzier (once it
stops bouncing, extra damping barely changes the trajectory — identifiability fades,
matching the gradient-magnitude fade in the cd sweep). Slight consistent
under-damping bias (recovered a touch bouncier than true), ~tracking-noise level.
`outputs/scene_compare.png` (height-vs-time + recovered-vs-true). Visual page
updated: https://claude.ai/code/artifact/2554f4fb-dfa7-4331-9b3a-860f77de8e20

## M15 — squishy splat with the cloth engine (2026-07-24)

`src/sim/diff_splat.py`, `scripts/render_splat.py`. A free (unpinned) cloth sheet
dropped TILTED onto the table, using Newton's SemiImplicit cloth solver (which IS
differentiable) + a custom differentiable ground penalty+friction force (same
pattern as the wind kernel). Stiffness (tri_ke) = squishiness.

- Stability needed tuning (heavy particles mass 0.04, substeps 48, penalty damping
  cd=25) — same explicit-cloth envelope as M0.
- Flat-on-flat gives no signal (both settle flat); TILTED drop makes it collapse/
  fold, and floppy vs stiff diverge ~10 cm in the transient (floppy conforms flat,
  stiff stays propped up — clear in `outputs/splat_compare.png`).
- **Gradient is unreliable** here: the extended cloth-ground contact (many particles
  landing over the collapse) breaks smoothness (tape != FD), same wall as rigid
  settling. **CEM (gradient-free) recovers stiffness to 0.1%** (ke 799.6 vs 800) —
  zeroth-order tolerates the rugged contact landscape, as throughout.
- Visuals: `splat.gif` (floppy cloth splatting), `splat_compare.png` (floppy vs
  stiff mid-collapse). Added to the visual page (rigid bounciness + soft squishiness):
  https://claude.ai/code/artifact/2554f4fb-dfa7-4331-9b3a-860f77de8e20

Confirms the material-from-motion thesis extends from rigid (bounciness/friction) to
soft (squishiness/stiffness), reusing the differentiable cloth engine from M0-M3.

## M16 — physics from a REAL Wan-generated video (2026-07-24)

`scripts/blender_ball_i0.py`, `scripts/recover_wan_ball.py`. First run of the full
pipeline on video that WASN'T our own render — Wan 2.2 TI2V-5B generates the motion.

- **Asset choice**: a bouncy ball, rendered in the same city/table scene as the single
  i2v conditioning frame. Rationale: video models animate FAMILIAR objects (balls) far
  more physically than an obscure scanned toy, and a ball's centroid tracks trivially by
  colour + "largest round blob". (Tried the triceratops I0 first — Wan handles the ball
  far better.)
- Generated 7 seeds. High-drop seed 0 is the only physics-rich take; low-drop seeds
  (ball just above table) barely move — no room to fall. Seed 5 flew upward (unphysical).
- **Tracking is the hard part**: the street scene has other pinkish round things and the
  shakers are the same salmon as the ball; loose colour tracking drifts. Fixed with a
  strict "largest round blob, area>3500" detector (24/49 frames valid; the gap is the
  impact where Wan deforms the ball out of round).
- **Result (honest)**: Wan's FALL is genuinely gravity-like — from the known ball radius
  (0.075 m) + 24 fps, recovered g = 8.9 px/f^2 -> **~9.8 m/s^2**, and a free-fall-from-rest
  model lands exactly with the tracked ball. But the SETTLING is non-physical: after
  impact the ball bobs ~66 px up and back down in place (constant x, constant apparent
  radius) — a real ball can't do that. So restitution isn't cleanly readable; the ball
  does not execute a clean decaying bounce.
- **This is the thesis in miniature**: you can't read physics straight off a generated
  video (the raw motion leaves the physical manifold once contact/settling starts). The
  fall is recoverable; the non-physical settle is exactly what the physics-projection
  discards. Added as the closing "real test" section of the visual page:
  https://claude.ai/code/artifact/2554f4fb-dfa7-4331-9b3a-860f77de8e20

Reproduce: blender_ball_i0.py (I0) -> run_i2v.py wan5b (video) -> recover_wan_ball.py
(track + fit + figures in outputs/wan_ball/).

## M17 — harden the Wan pipeline: robust tracking + 3D disentanglement (2026-07-27)

Weak links from M16 were (a) tracking (blob/LK drifted, lost 24/49 frames on impact
deformation) and (b) only gravity was recoverable. Hardened both.

- **CoTracker3** (`src/track/cotracker.py`, `scripts/track_wan_cotracker.py`): learned
  point tracker, installed into the `video` env. Seed query points inside the ball's
  frame-0 colour mask, follow them through blur/squash/occlusion. Result: **96% visible
  over 49/49 frames** vs 24/49 for the strict blob tracker. Reusable for real video.
- **Bounce-vs-roll diagnostic** (the headline finding): the ball's apparent SIZE (spread
  of tracked points = depth proxy) is a near-perfect mirror of its image HEIGHT
  (**corr = -0.98**). So the up-down in the image is the ball rolling toward/away from the
  camera on the table — **NOT a vertical bounce**. Better tracking doesn't just track
  better; it lets us diagnose the true 3D motion the monocular video disguises.
- **3D physics fit** (`src/sim/diff_sphere.py`, `scripts/recover_wan_ball3d.py`): a
  differentiable point-sphere (gravity + ground penalty restitution + Coulomb friction)
  is fit so its CAMERA-PROJECTED centre matches the CoTracker centroid (steps 6-7). The
  camera is validated (projected start [272,110] vs tracked [271,105], radius match).
  On the first physical arc (fall + roll to front) it recovers a small launch velocity,
  mu~0.4, and **cd pinned at its max (restitution ~0)** — independently corroborating the
  no-bounce diagnostic. Residual ~30 px: Wan's back-and-forth roll on a flat table is
  non-physical and a flat-table sim can't reproduce it (honest limit, not featured).

Net: the generated-video path is now robust to track and can DIAGNOSE 3D motion, not just
fit 1D height. Page updated (robust-tracking gif + rolling-vs-bouncing figure):
https://claude.ai/code/artifact/2554f4fb-dfa7-4331-9b3a-860f77de8e20

Env note: `cotracker` (from git) added to the `video` env; checkpoint scaled_offline.pth
cached under ~/.cache/torch/hub.

## M18 — add SANA-Video backend; robust CoTracker centroid (2026-07-27)

Added SANA-Video 2B (NVLabs/MIT, `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`)
as a selectable i2v backend (`src/video/i2v.py` -> `SanaI2V`, key "sana"). It's already
in our installed diffusers 0.38 (`SanaImageToVideoPipeline`) — no upgrade needed. Uses a
FlowMatch scheduler (flow_shift=8), motion set via a " motion score: N." prompt tag,
16 fps native. Note: `frames=` (not `num_frames=`), and `use_resolution_binning=False`
because the 480 bin remaps 448x544 -> 560x720 which fails the /32 check.

Empirical swap comparison (same high-ball I0, 448x544, 49 frames):
- **motion score matters a lot.** score=40 -> the scene degrades badly (garish noise band,
  camera drift despite the static-camera prompt, ball dissolves by ~f42). score=12 -> clean
  and stable: ball drops, lands, rolls toward the camera, background intact.
- **appearance drift breaks the visibility flag.** SANA shifts the ball salmon->saturated
  red and grows it a lot; CoTracker's positions still track it, but visibility collapses to
  7% (vs Wan 96%). Fix: centroid = median of visible points, FALL BACK to all points when
  <30% visible (`track_wan_cotracker.py`). Don't throw a track away on the vis flag alone.
- **rolling read-out holds for both** but Wan is cleaner: height–size corr = -0.98 (Wan,
  mean-of-visible) vs -0.23 (SANA, all-points). Two effects compound — SANA is harder to
  track AND its motion is less physically consistent.

Verdict: SANA-Video is a working, faster/smaller alternative, but for the physics pipeline
(which needs a trackable object with consistent appearance through the motion) **Wan 2.2 is
the better motion prior right now**. Kept both; page unchanged (the Wan -0.98 result stands).

## M19 — Cosmos3 world-model backend: first real bounce from generated video (2026-07-27)

User asked to try Cosmos 3 (SANA-Video 5B/"2.0" isn't released — only the 2B is public).
Cosmos3-Nano (`nvidia/Cosmos3-Nano`, ~33 GB omni world model) was already cached.

Integration: needs `Cosmos3OmniPipeline` (diffusers main, not our 0.38). Installed
diffusers-from-main + accelerate into the ISOLATED `cosmos` conda env (torch cu128,
transformers 5.14, vllm) so the working `video` env is untouched. Pipeline requires
`cosmos_guardrail`; bypassed with `enable_safety_checker=False`. Prompt is JSON scene
form. Added `CosmosI2V` backend (key "cosmos") in `src/video/i2v.py`; runs in the cosmos
env. Gen: 49 frames @448x544 in ~50-70 s (faster than Wan 84 s, SANA 170 s), fits one A6000.

Result — **the best motion prior yet, and the first with recoverable restitution:**
- **Stable + consistent**: static camera held, no scene degradation, ball keeps its
  appearance -> CoTracker **99% visible** (vs Wan 96%, SANA 7%).
- **A REAL vertical bounce in place**: apparent size (depth) varies only **±1%** (Wan/SANA
  ±6%, i.e. rolling) while height rebounds (fall -> up -> settle). This is the in-place
  bounce Wan/SANA never produced.
- **Restitution recovered**: 3D DiffSphere fit through the camera gives cd≈21 (moderately
  bouncy) — a real, non-zero restitution, vs the dead cd=40 forced by Wan's roll. Model-free
  rebound/impact-speed ratio ~0.3-0.5. Residual ~22 px (Cosmos's small settling wiggles).

Takeaway: the WORLD model (trained for physical plausibility) gives markedly more physical,
stable, trackable motion than the general video models — validating world-models as the
right i2v prior for this pipeline. Comparison figure: `three_priors.png`.

## M20 — settling pass for multi-object scenes (2026-07-27)

`src/sim/settle.py`, `scripts/settle_scene.py`. Relaxes a roughly-placed multi-object
scene (floating, interpenetrating assets) into a sim-ready resting state — the gating
"consistent initial configuration" item from the sim-readiness discussion.

Numerical journey (documented so we don't repeat it):
- **Explicit penalty dynamics FAILED** for settling-from-overlap: stiff springs + initial
  interpenetration eject objects (saw 30 rad/s spins, penetration growing). Velocity
  clamping just produced a jittering standoff. Penalty/explicit is the wrong tool here.
- **Root cause of non-convergence**: `sphere_cover` is a SURFACE SHELL, so two objects
  *resting in contact* already overlap by ~a sphere radius — the solver can't tell
  "touching" from "penetrating" and repels forever.
- **What works**: POSITION-BASED relaxation (projected depenetration + a gravity bias that
  ramps to 0), with (a) a summed, step-CLAMPED push (averaging diluted deep overlaps), and
  (b) a **contact tolerance** (~one sphere radius) so surface contact is an equilibrium and
  only true interpenetration is pushed. Unconditionally stable, converges monotonically.
- Result: rough scene (3 GSO assets floating 10-12 cm up, cores overlapping) -> **0.0 mm
  penetration, 0.000 mm/iter motion (fully at rest), each object dropped onto the table**.
  `outputs/settle_before_after.png`, `outputs/settle.gif`.

Scope/honesty: translation is relaxed, input orientations kept (assets placed upright);
object-object resting contact is at sphere-cover resolution. For finer/rotational settling
we'd want a solid (not shell) sphere fill or Newton's rigid solver — noted, not needed yet.

## M21 — "default vs recovered" deformation demo (2026-07-27)

`scripts/default_vs_recovered.py`. The "no default fits all" comparison (user request):
same tilted cloth drop, two fabrics (floppy ke≈250, stiff ke≈1400). One DEFAULT stiffness
(ke≈600) is applied to both vs the stiffness recovered per object.
- floppy: default is 8.8 cm off (too stiff, barely drapes); recovered matches (0 cm).
- stiff : default is 5.2 cm off (too soft, over-collapses); recovered matches (0 cm).
Targets chosen in the RESPONSIVE ke range (stiffness saturates >~1500, so 1500 vs 6000 look
alike — the first attempt accidentally near-fit the stiff case). Figure: 3 cols (observed /
one-size default / recovered) x 2 rows; the default column is identical in both rows — the
whole point. This is also our OBJECT-DEFORMATION story: stiffness is read from how the sheet
deforms as it collapses.

Deformation coverage (for the record): CLOTH / surface deformation = YES (this + M15 splat).
Volumetric soft SOLIDS (FEM jelly) = not built yet (in θ scope; Newton supports it) — the
natural next material if we want squishy solids.

## M22 — volumetric soft body (FEM jelly) (2026-07-27)

`src/sim/diff_soft.py`, `scripts/recover_soft.py`. Extends deformation recovery from
cloth (2D surface) to a 3D SOLID: Newton `add_soft_grid` (tetrahedral FEM, 4^3 cube,
320 tets) + SemiImplicit solver + the same differentiable ground-penalty. Recover the
FEM shear modulus k_mu (the 'squishiness') from how the cube squashes on impact.

- Stability: low k_mu + contact inverts tets and explodes. Fixed with k_damp=10,
  lam_ratio=2 (bulk stiffness resists inversion), 64 substeps.
- Discrimination: soft (k_mu 3k) squashes ~40%, stiff (k_mu 30k) ~6% — clear.
- **Observability lesson**: a gentle 2.5 cm drop barely distinguishes soft stiffnesses
  (loss flat, CEM recovered 56% off). A 6 cm drop makes k_mu sharply observable (loss
  minimum 1.8e-11 at truth). "You recover what the motion exercises" — again.
- Recovery: coarse log-grid seed + CEM refine (plain CEM collapsed into a shallow
  local basin) -> **k_mu recovered EXACTLY, 0.0% error** (3000 vs 3000).
- Videos: `outputs/soft/soft_jelly.gif` (jelly squashing), `soft_compare.png` (soft
  pancake vs stiff cube).

Deformation coverage now: cloth/surface (M15/M21) AND volumetric solid (M22).

## M23 — the probe→parameter matrix (the "scene as a lab") (2026-07-27)

`src/sim/probe_scene.py`, `scripts/probe_matrix.py`, `scripts/render_probes.py`.
One scene (teapot A + lion B on the table), three excitations, and a measurement of
what each experiment can actually identify. θ = {restitution cd, friction mu, density
ratio ρB/ρA}; truth {12.0, 0.35, 2.0}.

Two methodological traps found and fixed (both worth remembering):
1. **Soft penalty contact leaks a fake mass signal.** At k=4e3 a PARKED object's static
   sink depth depends on its density → 10 mm → **13 px** in the image, so "mass ratio"
   looked identifiable from a probe where B never moves. Fixed by stiff contact
   (k=4e4 → 0.4 px, dt=1e-4 for stability).
2. **Zero-noise identifiability is meaningless.** With a perfect model and no noise, any
   parameter that shifts the image by 0.01 px is "recoverable" — the first matrix came
   out all-green. Identifiability must be measured against a noise floor: we add 2 px
   tracking noise (CoTracker scale) and take the median recovery over 40 noise draws.
   Sims are precomputed per grid value so the noise averaging is free.

**Result (the matrix)** — effect on the image vs the 2 px floor:
| probe | restitution | friction | mass ratio |
|---|---|---|---|
| drop    | ✓ 19.0 px (8% err)  | ✓ 3.3 px (4%)   | ✗ **0.4 px** (invisible) |
| slide   | ✗ **0.1 px** (invisible) | ✓ 67.3 px (4%) | ✗ **0.4 px** (invisible) |
| collide | ✓ 95.7 px (7%)      | ✓ 232.8 px (4%) | ✓ **66.9 px (5%)** |

- The push never impacts → **bounciness is literally invisible (0.1 px)**.
- **Mass stays invisible until the objects collide** (0.4 px → 66.9 px) — the density
  gauge from M5, now demonstrated as an experiment-design result.
- The collision is the richest single probe (it identifies all three); the simple probes
  each see only a subset. Error floor ~4% is the recovery grid resolution, so the honest
  read is "identified (≈grid floor)" vs "unidentified (≫floor)"; the px effect column is
  the continuous evidence.

**Held-out prediction**: combine each parameter from the probe that identifies it, then
predict a FOURTH unseen probe (angled push into B) with no refitting →
**1.7 px error vs 24.5 px for default parameters (15x better)**. Fitting on one set of
experiments and predicting a different one is the strongest evidence the recovery is
physical rather than curve-fitting.

Outputs: `outputs/matrix/probe_matrix.png`, `heldout_prediction.png`, `probes.gif`.

## M24 — the probe→parameter matrix on REAL Cosmos-generated video (2026-07-27)

`scripts/blender_probe_i0.py`, `gen_probe_videos.py`, `src/track/balls.py`,
`scripts/cosmos_matrix.py`, `cosmos_matrix_figs.py`. The M23 matrix, repeated with the
targets replaced by video the Cosmos3 world model dreamed up.

Setup: two coloured balls on the real table, three probe initial frames sharing one
camera; Cosmos generates the motion (14 videos across probes/seeds); balls tracked by
colour; a two-ball rigid sim (ProbeScene `ball_radius` mode) is fit so its projected
centres match the tracks. **No ground truth exists**, so each cell reports the EFFECT
SIZE — how far the best achievable fit moves (px) when the parameter is swept, with the
unknown launch velocity re-fitted at every step — plus the resulting interval.

Findings, in order of value:
- **The headline result reproduces**: mass ratio is **invisible in drop and push
  (0.0 px)** and **strongly recovered only in the collision (72.8 px effect, ratio
  ≈ 1.41)**. The density gauge and the fact that a collision breaks it — first derived
  analytically (M5), then measured synthetically (M23) — now hold on generated video.
- **Friction is invisible in the drop (0.1 px)**: the ball falls straight down and never
  slides. Matches M23 exactly.
- **The pipeline REJECTS a non-physical probe.** The push video's path bends **57 px,
  17% of its length, off a straight line** — but a straight 3-D path must project to a
  straight image line, and a ball rolling on a flat table has no lateral force. So that
  motion is physically impossible; the fit refuses it (34 px residual, every parameter
  flat). This is architecture step 8 ("reject high-residual observations") firing on
  real generated video, and it is a capability, not a failure.
- **Most generated collisions are fake.** Causality screening (does blue only move once
  red actually touches it?) rejected 4 of 6 collide seeds — in those, the target ball
  accelerates with nothing touching it. Only seed 2 is causal (−3 px surface gap at the
  moment blue starts moving).
- Restitution/friction from the collision only give one-sided bounds (they rail at the
  prior edge); reported as bounds, never as recovered values.

Methods notes: the ball tracker had to be calibrated from the renders — the wooden
table is *also* reddish (r−g, r−b both large) so a naive red mask swallowed the ball
into the tabletop; separation is by brightness/blueness, plus the street HDRI contains a
red car so detection is limited to the table region. Camera validated on generated
video: projected start [260,169] vs tracked [261,167].

## M25 — joint recovery + held-out prediction on generated video; the lab planner (2026-07-28)

### Joint recovery (the "system of equations")
`scripts/cosmos_joint_heldout.py`. Every probe video of the same objects shares one
MATERIAL but has its own unknown launch velocity, so we solve one joint problem:
shared θ=(cd, mu, mass ratio) + per-probe nuisance v0_p, minimising the summed projected
residual over probes. 7 unknowns over {drop-s0, collide-s2} (push excluded: non-physical).
CEM converged 49.1 → ~15 px. **Jointly recovered: cd=7.26, mu=0.192, mass ratio=1.162**
(per-probe residuals 17.0 / 14.7 px).

### Held-out prediction on generated video — INCONCLUSIVE, and we can say exactly why
Freeze the material, fit ONLY the held-out clip's launch velocity (an initial condition
cannot transfer, a material must), compare against default material.
- **Attempt 1** (`heldout` layout, seed2): 13.2 px both ways, 1.00x. Diagnosis: the
  clip's knock is a glancing 18 px tap AND the fitted trajectory never even brings the
  balls into contact — the two candidate materials predicted the same thing to within
  2.9 px (target ball: 0.0 px). A vacuous test, not a failed recovery.
- **Attempt 2** (`collide` seed4 — strong causal knock, target moves 255 px; contact does
  happen in the sim, and the two materials' predictions differ by 9.8 px, so the test is
  live): transferred **26.3 px** vs default **27.1 px** — only 1.03x.
- **The control that settles it**: fitting material *directly on the held-out clip* gives
  **26.5 px** — the best the model can possibly do. So the transferred material is already
  optimal for this clip (26.3 ≤ 26.5); the ~26 px floor is the VIDEO's own departure from
  physics, not parameter error, and it swamps the ~10 px of material signal.

**Conclusion, quantified**: held-out transfer is cleanly demonstrated in simulation
(1.7 px vs 24.5 px = 15x, M23) but NOT yet on Cosmos video. The bar this sets is concrete:
generated motion must be physically consistent to better than ~10 px (the material-induced
prediction spread) before transfer can be demonstrated on it. Cosmos collisions sit at ~26 px.

### The lab planner (`src/lab/planner.py`)
Turns a requested attribute into the experiment that reveals it — "I want this duck to be
squishy" -> soft_stiffness -> hard_drop probe + prompt + setup, justified by the measured
6 cm-drop (0.0% error) vs 2.5 cm-drop (56% error) result. Contains: attribute aliases
(bouncy/slippery/heavy/squishy/floppy) -> parameter -> probe; the measured effect-size
table; `screen_take()` with the physical-validity gates measured on generated video
(tracking coverage, path curvature >10% = non-physical, contact causality, fit residual);
and an escalation ladder for when the signal is too weak.

## M26 — scene milestone (part 1): assets, scene, SimReady write-back, video-vs-sim (2026-07-28)

Pivot to the pitch's actual deliverable: a whole authored SCENE handed back with physics.

**Assets** (`scripts/fetch_props.py`) — 9 CC0 Poly Haven props chosen so they look like
ordinary dressing but must move very differently: rubber_duck_toy (the pitch's own duck),
throw_pillows_01, baseball_01, food_apple_01, ceramic_vase_01, brass_pot_01,
cardboard_box_01, book_encyclopedia_set_01, wooden_bowl_01. All load in trimesh at real
scale. (coffee_cart 404s.) Poly Haven's API needs a User-Agent or it 403s.

**Scene** (`scripts/blender_scene.py`) — a finished-looking tabletop: duck, brass pot,
ceramic vase, wooden bowl, baseball, apple on the wooden table under the city HDRI.
The pot/vase pair is deliberate: same vessel silhouette, one dense metal and one thin
ceramic — they cannot share a default. Renders `outputs/scene/hero.png`, writes
`scene.json` (camera, table height, per-object placement + bbox), and exports
`scene_geom.usdc` via Blender's USD exporter (Z-up, metres, prims at /root/<name>).

**SimReady write-back** (`src/simready/usd_physics.py`) — installed `usd-core` (USD 26.8;
schema names verified against the installed lib, not memory). Per movable object applies
UsdPhysics **RigidBodyAPI + CollisionAPI + MassAPI(density)** and binds a
UsdPhysics **MaterialAPI** carrying dynamicFriction / staticFriction / restitution;
static prims (table, floor) get CollisionAPI only; a UsdPhysics.Scene carries gravity.
Round-trip verified by reading the attributes back off the exported stage.
Every value also carries **provenance + confidence** custom attributes
(`simready:provenance:<param>` = "recovered:<probe>" or "default:class-prior"), and
`coverage_report()` prints which of an object's numbers are real. NOTE: the values
exercised so far are PLACEHOLDERS — per-object recovery for these props is the next step.

**Video vs simulation** (`scripts/video_vs_sim.py`) — side-by-side of the Cosmos clip and
our simulator running the material fitted from it, same camera and backdrop, for the drop
and collide probes. (Frames rendered explicitly; FuncAnimation+PillowWriter sheared them.)

## M27 — per-object recovery on the authored scene; screening hardened (2026-07-28)

Full loop run per scene object: lift the object in its own scene -> Cosmos drop video
(24 clips over 5 objects) -> CoTracker -> physical screening -> fit -> SimReady USD.

**Result, honestly: 1 of 7 objects carries a measured value.** The apple's restitution
recovers cleanly (e=0.361, from cd=7.5, 11.3 px fit). The other four probed objects fail
the credibility gate with residuals 28-37 px; the two unprobed ones keep class priors.
All of this is recorded in the asset as provenance/confidence, so nothing is silently
guessed. Note the coincidence that matters: our credibility threshold (25 px) sits right
at the ~26 px physical inconsistency we measured for Cosmos clips in M25 — most objects
fail because the generated motion is only ~30 px consistent with ANY physics, not because
the fit is broken.

Bugs found and fixed along the way (each was silently producing wrong answers):
- **Tracker seeded on the wall.** Seed radius used the object's LARGEST dimension, so a
  tall thin vase (12.6 cm wide, 24.8 cm tall) got a seed circle full of background and
  CoTracker followed the wall — the vase looked like it "barely fell". Now uses the
  smallest horizontal extent.
- **A 233 px fit was reported as "identified".** A large scan spread is not identification
  if the simulator cannot reproduce the motion at all; added a fit-credibility gate.
- **Equivalent-sphere modelling.** A vase and a duck do not land like a ball. Now uses the
  real mesh sphere-covered at its as-placed scale (ProbeScene gained `mesh_scale`, and
  accepts "category/name" so soft/ assets load).
- **Mismatched reference frames** in the relative-motion loss (sim indexed from frame 0,
  observation from the first *tracked* frame).
- **Mesh covers need far softer contact**: hundreds of spheres each pushing at k=4e4
  launched the body off the table; k=2500 with 80 substeps is stable.
- **Take selection picked the biggest fall**, i.e. the most anomalous clip (a vase
  "falling" 218 px when only 85 px was possible). Now picks the fall closest to the one
  we staged, and rejects falls beyond 1.7x the staged drop.

Restitution is written to USD as a real coefficient by running the recovered sim and
measuring rebound/impact speed, rather than inventing a cd->e mapping.

## M28 — motion-signature loss (2026-07-28)

`src/motion/signature.py`. Replaces the pixel-trajectory objective with a small set of
scale-free features of the motion: **rebound_fraction** (how high it comes back relative
to how far it fell — essentially restitution), **settle_frac** (decay envelope), and
**fall_frac** (when the landing happens). These are invariant to the things we neither
control nor care about — a constant offset between the tracker's centroid and the sim's
body origin, small timing shifts, lateral drift, modest scale error — which is precisely
the ~30 px of pixel disagreement that was failing four of five objects.

**A/B on the same clips and tracks — objects whose motion the simulator can explain at all:**
    pixel loss     1/5
    signature loss 5/5
That is the barrier it was meant to remove, and it removed it.

What remains is a different and more honest limit: **the generated drops barely bounce**
(observed rebound 0-17%, mostly ~3%), so restitution is weakly observable no matter the
loss. Identification is now gated on the interval of damping values that fit within
tolerance — a plateau ("any value over 4-106 fits equally", the apple) is reported as a
bound, not a measurement. Final: **2/7 objects carry a measured value** —
brass_pot e=0.095 (interval [8-15], confidence 0.88) and baseball e=0.010 ([55-206], 0.75).

Note worth keeping: **the recovered baseball is dead (e=0.01)**, which is wrong for a real
baseball and right for the video we were given — Cosmos generated a baseball that does not
bounce. The pipeline faithfully recovered the physics it was shown. That is the correct
behaviour and it locates the error where it belongs: in the instrument, not the estimator.

Gating fixes made at the same time: failed/unstable rollouts (1e6 sentinels) no longer
pollute the spread statistic, and identification requires the near-optimal interval to be
a small fraction of the scanned range rather than merely "spread > noise".

## M29 — escalation: what actually moved the needle (2026-07-28)

Ran the planner's escalation ladder on the failing scene objects. Three passes, same
objects/scene/fitting code:
  v1  "falls, lands and settles", 16 cm drop, tight camera
  v2  "visibly bounces", 30 cm drop, camera re-framed for headroom
  v3  "visibly bounces", 16 cm drop, tight camera   (prompt isolated)

**The escalation itself was essentially a null.** Mean rebound delivered by Cosmos:
v1 7.9% -> v3 8.8% (prompt alone), and v2's bigger drop gave 9% while dropping mean
tracking coverage 71% -> 51% (the re-framed camera makes objects smaller and faster).
Neither asking for a bounce nor doubling the drop height controls how much the model
rebounds. Individual takes DO reach 34-39% rebound, so the information exists — the job
is selecting those takes, not commanding them.

**What actually moved the needle was fixing our own screening** (see M29 commit): the
"barely falls" gate, written for the pixel loss, was discarding exactly the informative
clips because the signature is normalised by the ACTUAL drop. On identical v1 clips with
no new video, identified went 1/5 -> 3/5. Pooling all compatible takes (v1+v3, 39 tracks)
gives **5/5 identified**, each with a tight damping interval.

**Final asset: 4/7 objects carry a measured restitution**, and three of the four are
physically right:
    baseball      e=0.505   (real baseball on a hard surface is ~0.5)   plausible
    brass_pot     e=0.099   (heavy metal barely bounces)                plausible
    rubber_duck   e=0.770   (hollow rubber)                             plausible
    ceramic_vase  e=0.980   SUSPECT — a ceramic vase does not bounce like a superball
The vase failure is diagnosable and worth fixing properly: **the drop signature assumes
bouncing, not toppling.** A tall object that tips over raises its centroid, which the
rebound feature misreads as a bounce. Next fix is to detect toppling (orientation change)
and either reject the take or use a different signature for tall objects.
Also open: the apple is identified but `measure_restitution` returns None for it, so it
silently falls back to a prior — a gap that is logged but not yet chased.

The "dead baseball" of M28 is fully explained and gone: it was our prompt telling the
model not to bounce AND our screen discarding the takes that did.

## M30 — joint fitting, and the finding that matters: the model is not repeatable (2026-07-28)

Two things asked for: play the clips at full frame rate, and optimise jointly.

**Frame rate** was our GIFs, not the videos: source is 49 frames @24fps, our GIFs showed
17 frames @7fps (every 3rd frame). Fixed to play every frame at 24fps. The residual
limit is real but smaller: a 0.3 s bounce is only ~7 source frames.

**Joint fitting across takes made things worse** — 5/5 identified -> 1/5 — and chasing
why produced the most important measurement of the project:

    same object, same prompt, only the seed differs; rebound as % of the fall
      baseball      0, 0, 0, 0, 6, 26, 32, 34      (34 point spread)
      apple         0, 0, 0, 3, 8, 20              (20)
      brass_pot     0, 3, 6, 7, 9, 9               (9)
      ceramic_vase  0, 0, 0, 2, 4, 4, 15, 17, 39   (39)
      rubber_duck   0, 0, 0, 1, 2, 3, 13, 17, 28   (28)
    mean spread 26 percentage points

**Cosmos is not repeatable.** No single material can explain both a 0% and a 34% rebound,
which is exactly why a pooled objective fits none of them. So the estimator was replaced
with a CONSENSUS test: fit each take on its own, then ask whether the takes agree; the
agreement IS the uncertainty. Result: only the baseball's takes agree (within 1.5x);
apple's disagree by 5x and the vase's by 8x.

**This retracts M29's headline.** The "4/7 objects, 3 of 4 matching textbook values"
result came from selecting the bounciest take per object — cherry-picking. Under a
reproducibility test those values do not survive: the baseball's 0.505 was reproducible,
the others were selection artefacts. The right claim today is: the pipeline works and
can now MEASURE its own reliability, and by that measure generated video is not yet a
repeatable instrument for per-object material recovery.

Constructive read: the information is present (takes reaching 26-39% rebound exist), it
is just not reproducible on demand. Paths forward are more seeds with consensus (costly
but works — the baseball did converge), a more deterministic conditioning, or a video
model whose physics is stable across seeds. This is a measurable acceptance criterion to
hold future models to.

## M31 — joint recovery across DIFFERENT experiments (2026-07-28)

The right reading of "optimise jointly": not pooling repeats of one probe, but solving
several *different* experiments for one parameter set, since each reveals a different
parameter and together they constrain each other.

Staged in the authored scene (`blender_scene_lab.py`), baseball as mover, apple as target,
other props parked clear of the runway:
    drop -> restitution, slide -> friction, collide -> mass ratio

Two prerequisites had to be built first:
- **the friction channel.** The motion signature was vertical-only (rebound, settle), so
  friction was invisible to the objective and no amount of joint fitting could have found
  it. Added `slide_frac`, plus `slide_signature` (travel in object-widths, deceleration
  ratio, stop time) and `collide_signature` (momentum transfer, mover's retained travel).
  Signal is real: brass pot slides 0-3% of its drop, baseball up to 300%.
- **a collision plausibility gate.** A struck object cannot leave an impact with more
  motion than it arrived with. All three collide takes failed it — the best-tracked one
  had the ball travelling 1.77x further AFTER the hit while moving the apple by 0.08x.

**Result** (`lab_joint_fit.py`, CEM over 3 shared parameters + 2 per-clip launch velocities):
    joint cost 0.232, experiments used: drop + slide, rejected: collide
    restitution damping = 135.7   DETERMINED       (cost moves 0.381 when perturbed)
    friction mu         =   0.548 DETERMINED       (cost moves 0.033 — weakly)
    mass ratio          =   0.150 NOT CONSTRAINED  (cost moves 0.000)

The mass ratio result is the point: with the collision rejected there is nothing left in
the data that can see mass, and the fit says so instead of returning the number it drifted
to. A post-fit sensitivity test decides this rather than assertion — perturb each parameter
around the optimum and see whether the joint cost notices.

This is the pitch's "system of equations" working end to end on authored scene objects,
including the part that matters most: it reports which parameters the experiment set
actually determined.

## M32 — making the collision work; joint vs sequential (2026-07-28)

**Why the collision failed — it was not the joint fit.** The rejecting gate runs on the
observed tracks alone, before any simulation. Two real causes:
1. *No collision was generated.* The target's tracked position across all takes was
   269->272, 269->269, 269->269 px — it never moved. The objects were staged 31 cm apart
   and the mover never arrived within the 2 s clip. (The two-ball scene that did collide
   had them ~11 cm apart.) Restaged at 11 cm.
2. *The tracker lost the mover.* The white baseball tracked at 4/53/11%; in one take
   CoTracker followed it to x=1027 in a 544-wide frame. The red apple tracked at 97-99%
   in the same clips, so the roles were swapped: apple = mover, baseball = target.
   My earlier claim that the clip was unphysical was overstated — I was measuring a
   tracker that had wandered off, and should have checked track quality first.

**A geometry bug was corrupting every signature.** The body centre was derived from the
mesh bounds, but ProbeScene centres its sphere cover on the mesh's vertex mean and the
cover is a voxelised shell of finite-radius spheres. Objects therefore started buried and
popped up ~6 cm on frame 1. Now the rest height is computed from the actual cover
(`ground_z + r - min(centre_z)`); vertical drift for slide/collide is 0.0 cm and the drop
falls exactly the 20 cm staged.

**Both estimators now run over all three experiments, none rejected:**

                    sequential          joint
    restitution     53.8  not constr.   21.6  not constrained
    friction         0.740 DETERMINED    0.171 DETERMINED
    mass ratio       1.630 DETERMINED    0.670 DETERMINED

They AGREE on *which* parameters the experiment set determines (friction and mass yes,
restitution no) — mutual validation of the identifiability structure. They DISAGREE on the
values by 2-4x, which is the honest uncertainty: larger than either fit's own sensitivity
test suggests.

**Joint beats sequential where it counts.** True density ratio baseball/apple is ~0.89
(baseball ~0.74 g/cm3, apple ~0.83). Joint recovers 0.67, sequential 1.63. The likely
reason is exactly the argument for joint fitting: sequential froze a bad restitution (from
a drop take whose 86% rebound is implausible) and propagated it into the collision stage,
while the joint fit could trade it off. Sequential remains the better debugging tool;
joint is the better estimator.

Open: the drop signature needs a plausibility gate of its own (an 86% rebound for an apple
should be rejected the way an accelerating collision now is), which is likely why
restitution is unconstrained in both fits.

## M33 — drop gate, scene-wide lab, repeatability harness (2026-07-29)

All three next steps.

**Drop plausibility gate** (`src/motion/signature.py`). Rebound/fall is e^2, so >0.80 needs
e>0.89 — a superball. Plus: a real bounce must come back DOWN (a trace ending at its peak
was still rising), and a fall under 20 px makes the ratio noise. Rejects 5 of 43 existing
takes: calibrated rather than blunt, and it removes the 86% apple rebound that was
poisoning the earlier restitution fit.

**Scene-wide lab** (`blender_full_lab.py`, `gen_full_lab.py`): 15 experiments =
drop/slide/collide for all five probeable objects, 45 clips, every collision against the
SAME reference object so mass ratios share one scale. All 45 generated and tracked.

**Repeatability harness** (`full_lab_fit.py`): per-take fits with seed agreement as the
uncertainty; a value the seeds disagree about by >4x is "not established", a value from a
single take is "unverified".

    object         restitution                       friction                      mass
    apple          5.10  1 take, unverified          1.100 RAILED at grid max      no take
    baseball       58.8  3 takes, 11.5x -> refused   0.680 1 take, unverified      no take
    brass_pot      no usable take                    0.365 2 takes, 1.7x ESTAB.    no take
    ceramic_vase   132.8 1 take, unverified          0.470 2 takes, 1.0x ESTAB.    1.00 1 take
    rubber_duck    nothing usable at all

**Final asset: 6 measured values across 4/7 objects.** The three genuinely established
values are all FRICTION, and all three are physically plausible:
    brass_pot 0.365 (metal on wood ~0.3-0.5), ceramic_vase 0.470 (ceramic ~0.4-0.6),
    baseball 0.680 (leather ~0.4-0.6, at the high edge)

Honest reading:
- **Friction is the reliable channel.** The slide gives 3/3 usable takes on every object and
  the tightest seed agreement in the project (1.7x). Sliding is slow and sustained, which is
  what a 24 fps generated clip represents well.
- **Restitution is not established anywhere.** The baseball had three usable takes that
  disagree by 11.5x. Before the consensus test this would have been reported as a confident
  number by picking one take — the harness is catching exactly the error made in M29.
- **Mass is established nowhere**: no usable collision for any object even at an 11 cm gap.
  It is the one parameter with no substitute experiment.
- The apple's friction railed at the grid maximum (1.1) and is downgraded rather than
  written — a bound is not a measurement.
- The rubber duck failed every experiment; the ceramic vase's restitution (0.98) is still
  the toppling artefact flagged in M29 and should not be trusted.

## M34 — mass established; the failure was error propagation, not the collision data

Diagnosed why every mass measurement failed. It was NOT the collisions: 7 of 15 takes have
good collision signatures, the apple's being textbook (approach 1.2 object-widths,
transfer 0.86, mover keeps 0.04). The failure was a cascade:

    apple friction railed at the grid maximum (1.10)
      -> the collision stage INHERITED it
      -> at mu=1.10 the simulated mover decelerates so hard it never reaches the target
      -> no collision in sim at any launch speed searched
      -> every candidate mass ratio scores identically badly -> "no usable take"

Verified directly: at mu=1.10 the sim produces no collision at any velocity; at mu=0.40 it
collides at v0=1.4 and 2.0 m/s.

Fixes: (1) a stage may only inherit a value that was actually ESTABLISHED — an unreliable
or railed value is replaced by a neutral default; (2) the launch-speed search was widened
so the mover can arrive; (3) railed mass ratios are downgraded to bounds the same way
railed frictions already were (the duck came back at exactly the grid minimum).

**Result — mass is established for the first time:**
    ceramic_vase   ratio 3.82 (baseball density / vase density), 2 takes, seeds agree 1.7x
                   -> vase density written as 178 kg/m3
    Sanity: a baseball is ~740 kg/m3 and a HOLLOW ceramic vessel's effective density over
    its volume is a few hundred, so a ratio above 1 is the right direction and the
    magnitude is the right order.
    baseball  ratio 1.00 but seeds disagree 8.5x -> not established
    duck      railed at the grid minimum -> downgraded to a bound
    apple, brass_pot  still no usable collision

The ceramic vase now carries all three parameters measured. Final asset: 6 measured values
across 4/7 objects.

General lesson, now encoded: **propagate the STATUS of a value, not just the value.** Every
stage already computed whether its result was established; nothing was honouring that
between stages, so a bad number travelled silently and surfaced two stages later disguised
as missing data.

## M35 — remaining collision diagnostic, and a limit on what "established" means

Diagnosed each remaining failure precisely. Three different causes, none of them the data:
- **apple**: the sim CAN reproduce the observed collision (its transfer range 0.18-4.00
  covers the observed 0.86) but scored 0.404 against a 0.30 gate almost entirely on
  IMPACT TIMING — which is set by the launch velocity, a per-clip nuisance, not by the
  material. Weighting a nuisance at 0.4 let it vote on whether a measurement was credible.
  Down-weighted to 0.12.
- **brass_pot**: observed transfer 0.98, but the sim could only reach 0.60 anywhere on the
  mass grid — for a heavy mover to hand over that much motion the target must be lighter
  than the grid minimum. Grid extended 0.2 -> 0.06. (It also collided in only 3/35 cells vs
  the apple's 19, hence a finer/faster launch grid.)
- **baseball**: genuinely marginal (0.234 accept / 0.350 reject) — real borderline data.

**Result after the fixes: baseball's mass establishes (2.156, 2 takes agreeing) — but the
ceramic vase LOST its established status (3.82 -> 2.785, agreement 1.7x -> 4.6x).**

That trade is the important finding, and it is a caution rather than a win:

    object          impact_frac=0.40        impact_frac=0.12+wider grid   change
    ceramic_vase    3.82  ESTABLISHED       2.79  not established         1.4x
    baseball        1.00  not established   2.16  ESTABLISHED             2.2x
    rubber_duck     0.20  railed            3.60  1 take                 18.0x

**Two objects changed established-status from an objective-weight change alone.** So the
seed-agreement figure is NOT the whole uncertainty: model/objective choice moves these
values by 1.4-2.2x, comparable to or larger than the seed spread we were reporting as the
error bar. Physical check: the apple/baseball density ratio is really ~1.12; we recover
2.16 here and 1.49 from the earlier pair-lab — right order, about 2x out either way.

Conclusion: **mass should be reported with a factor-~2 band, not as a point value**, and
the confidence machinery needs a second axis (objective/model sensitivity) alongside seed
agreement. Both of my collision "fixes" were legitimate bug fixes, but the fact that
which objects pass keeps changing under them is itself the signal that these values are
not yet solid. Chasing further objective tweaks until the numbers look right would be
cherry-picking by another name, so stopping here and reporting the band.

## M36 — objective sensitivity as a second confidence axis (2026-07-29)

Every value is now judged on TWO axes, not one:
    seeds      does the answer survive a different random seed?
    objective  does it survive a different reasonable way of scoring the fit?
A value must pass both. Implemented cheaply by simulating once per grid point and caching
the SIGNATURE, so re-scoring under each of three objective variants (balanced /
timing-heavy / amplitude-only) costs nothing extra.

**It immediately caught the value I was about to report last turn.** The baseball's mass
ratio has PERFECT seed agreement (1.0x) but moves 7.7x depending only on how the fit is
scored — so it is now correctly rejected. Seed agreement alone would have called it
established. That is exactly the failure mode flagged in M35, now detected automatically
rather than by hand.

Final state — established values (pass both axes):
    brass_pot    friction 0.365   seeds 1.7x  objective 2.3x
    ceramic_vase friction 0.470   seeds 1.0x  objective 1.0x   (cleanest value in the project)
Everything else is single-take, seed-disagreeing, objective-sensitive, or railed.
Asset: 7 measured values across 5/7 objects, each carrying both spreads.

**Mass is established nowhere once objective sensitivity is counted** — the earlier
"established" mass readings (M34's vase 3.82, M35's baseball 2.156) both fail the second
axis. That is the honest position and it supersedes both.

Caveat worth remembering: **railing masquerades as agreement.** The apple's friction sits
at the grid maximum under all three objectives, so its objective spread reads a perfect
1.0x. Agreement metrics cannot detect a boundary optimum; only the explicit railed-check
catches it (and does, downgrading it before it is written).

## M38 — reference friction established; mass still not, and why

Generated 8 more baseball slide clips (3 -> 11), because every collision uses the baseball
as the reference and its friction governs how far the struck object travels in EVERY mass
measurement.

**The reference measurement succeeded on its own terms:**
    baseball friction 0.680 from 11 usable takes, seeds 1.6x, objective 1.0x -> ESTABLISHED
Four objects now have established friction (apple's rails at the grid max and is
downgraded before writing): baseball 0.680, brass_pot 0.365, ceramic_vase 0.470.

**But it did not unblock mass — and the way it failed is the most useful part.** The
baseball's mass ratio moved from 2.156 to 0.465 purely because the friction estimate
improved: a **4.6x shift in the mass answer from fixing a different parameter**. That is
direct confirmation that the confound measured in M37 is real and severe, and that every
earlier mass number was contaminated by a bad friction estimate.

Mass is established nowhere. The binding constraint has now MOVED: it is no longer a bad
friction, it is **too few usable collision takes** (apple 1, baseball 1, vase's 2 disagree
by 4.6x, pot and duck none). The slide went from 1 usable take to 11 by generating 8 more
clips; collisions have had no equivalent treatment.

Next lever, in order:
1. many more collision seeds per object — the same treatment that worked for the slide
2. a faster impact, to raise SNR on the velocity jump
3. >=60 fps video — the actual cure, unavailable with Cosmos at 24 fps

Requirements this project can now hand a video-model vendor, both derived from measurement
rather than opinion:
    >= 60 fps          (a 3-5 frame momentum transfer cannot be resolved at 24)
    <= ~10 px physical inconsistency  (from the held-out test in M25)

---

## M39 — the estimator was manufacturing numbers; rebuilt it as direct measurement

Earlier entries reference `src/motion/signature.py`, `scripts/full_lab_fit.py`,
`scripts/lab_joint_fit.py` and `scripts/recover_scene_objects.py`. **These are deleted.**
The history above is left as written; this entry records why they went and what replaced
them.

**What went wrong.** Asked to check whether mass finally established, we instead looked at
the clips. Over 55 collision takes the ceramic vase's subject never displaced more than
21 px and the rubber duck never more than 17 px, against a gap of ~23-43 px they had to
close. **The videos did not contain the experiment.** The old acceptance test missed this
because it measured PATH LENGTH, which integrates |dx| every frame and so accumulates
tracking jitter without bound — 49 frames of +-2 px noise manufactures ~100 px of "travel"
from a stationary object. Net displacement does not accumulate. An audit on that basis
found **21 of 47 "usable" takes were measuring nothing.**

Two further defects, both predating any video:
  * the brass pot **overlaps its partner in the initial frame** (60.6 px apart, contact at
    62.3 px), so that collision had no approach to observe and was invalid as staged;
  * `min_travel` as a single global constant is the wrong instrument — the travel needed to
    reach the partner ranges 0.00 to 1.05 object-widths across our own five objects.

**Why the whole approach was wrong, not just the gate.** The old fitter searched a grid for
the theta whose hand-weighted signature best matched, and a grid always returns a best
match; it never asks whether the observation constrains anything. Direct measurement asks
first. On the SAME 10 baseball slide takes, the individual takes give mu = 0.007, 0.015,
0.020, 0.043, 0.071, 0.079, 0.090, 0.154, 0.268, 0.285 — and the grid fitter reported
**0.680 "established"**, a value outside the entire range any single take supports. It also
reported the apple at exactly 1.100, the grid maximum, as ESTABLISHED: a railed bound
presented as a measurement.

**The replacement** (`src/motion/observables.py`, `scripts/simple_fit.py`, 842 -> 512
lines). Each probe is analytically invertible, so nothing is searched:

    drop     e   = |v_up| / |v_down|                          dimensionless
    slide    mu  = |a| / g                                    needs the px<->m scale
    collide  m_target/m_mover = (v_pre - v_post) / v_target    dimensionless

Every quantity is a velocity from a least-squares fit over ~5 frames, which yields a
standard error for free. Consequences:
  * gone: `CD_GRID`, `MU_GRID`, `RATIO_GRID`, `OBJECTIVE_VARIANTS`, `COLLIDE_WEIGHTS`,
    `SLIDE_WEIGHTS`, `AGREE_MAX`, `MIN_MOVER_TRAVEL`, `MIN_APPROACH_PX`. No tunables.
  * the three scoring objectives existed only because no single loss was ever justified;
    they measured our own arbitrariness and reported it as physics uncertainty.
  * "did it move" becomes a **significance test** on the fitted velocity — one principle,
    no thresholds in object-widths.
  * "no bounce" becomes a measurement (e = 0 +- sigma) instead of a rejected take.
  * `established / unverified / not-established` is gone: an interval spanning the
    plausible range already says it.
  * takes outside the physically possible range (a mover that SPEEDS UP through impact,
    implying negative mass) are reported and excluded, never averaged. 3 of the brass
    pot's 5 collisions do this; averaging them in previously gave m_t/m_m = -0.285.

**Result: 0 of 15 parameters measured with an interval tighter than +-25% from more than
one take.** Nothing in the scene is currently measurable from this video.

| object | restitution | friction | mass ratio |
|---|---|---|---|
| apple | — | 0.01 (1 take) | — |
| baseball | 0.00 +- 0.04 | 0.01 +- 0.03 | — |
| brass pot | 0.05 +- 0.08 | — | 0.51 +- 0.47 |
| ceramic vase | — | 0.01 (1 take) | — |
| rubber duck | — | — | — |

**A probe-design error this exposed.** The mu ~ 0.01-0.09 readings are about right for
**rolling resistance** of a ball on wood — which is not the Coulomb mu the simulator
consumes. A sphere's deceleration cannot measure sliding friction. The slide probe is valid
only for objects that actually slide (book, box), not the baseball or apple.

**Retractions.** The M38 claim that 60 fps was "the diagnosis" for mass was wrong: the
per-take fits were individually decisive (12x and 23x contrast) and mutually contradictory,
which is generator non-repeatability, not temporal resolution. Frame rate cannot be the
binding constraint while half the clips contain no motion. Also retracted: the "true"
densities used to judge error were a hardcoded `PRIOR` dict of textbook guesses, anchored
by `REF_DENSITY = 680` taken from that same dict — downloaded GLTF meshes have no true
density, and there is no ground truth anywhere in the generated-video path.

**Still load-bearing, extracted rather than deleted:** `src/lab/staging.py` (`obj_geom`,
contact constants) and `scripts/prepare_lab.py` (`seeds` and `track` subcommands, the
CoTracker pass). `scripts/inspect_collide_clips.py` was ported to the new observables and
remains the fastest way to see whether a clip contains its experiment.

**Next lever is data collection, not estimation.** Screen the generator before designing
probes: for each candidate, generate a few clips and measure *did the intended thing move*.
Static-equilibrium probes (flotation for absolute density against water, a balance beam for
mass ratio) remove both the frame-rate sensitivity and the arbitrary density anchor — but
they assume the model animates the object at all, which for the vase and duck it does not.

---

## M40 — screening the generator, and a correction to M39's central number

**M39 claimed "68 of 93 takes contained no measurable motion". That is withdrawn.**
An independent instrument puts it at **8 of 93**. The motion was there; the tracker was
not.

**The screen.** Before designing new probes we screened the generator: same requested
motion, four stagings, 36 clips (`scripts/blender_screen.py`, `gen_screen.py`). The
hypothesis was that the initial frame must depict NON-EQUILIBRIUM -- drops worked
(object staged airborne, falling is inevitable from the pixels) while slides and
collisions did not (object at rest on a table, nothing implies motion). `rest` and
`urgent` shared a byte-identical frame so any difference would be purely the prompt.

    equilibrium      (rest, urgent)      18/18 clips moved
    non-equilibrium  (airborne, tipped)  18/18 clips moved

**Hypothesis refuted.** Staging the object mid-air changes nothing, and every one of the
36 clips shows substantial pixel change. The ceramic vase changes 9.5% of its pixels
against the baseball's 1.3% while barely translating: the model RE-RENDERS the object
rather than moving it rigidly. Cosmos is not refusing to animate.

**Then the screen's own instrument failed.** It measured CoTracker centroids. 11 of 12
baseball clips had points off-frame most of the time, and a duck clip in which the duck
visibly falls to the table reported 6 px of motion because the points stayed in mid-air.
Point visibility cannot detect this -- the points really are visible, just not on the
object -- so the failure is invisible in the tracker's own diagnostics. 38% of the main
lab's 93 takes fail the same health check.

**Re-audit with an appearance-based instrument** (`src/motion/patch_track.py`, NCC
against the object's own frame-0 patch; `scripts/audit_pixel.py`). It reports both WHERE
the patch matches and HOW WELL, and the second number is what centroid tracking never
had:

    MOVES     71/93  (76%)   displacement with the asset intact
    STATIC     8/93  ( 9%)   genuinely nothing happened
    DEGRADED  14/93  (15%)   the object stopped being the object

**New failure mode: asset degradation.** All 14 DEGRADED takes are the brass pot, which
transforms mid-clip from a lidded pot into a wide shallow bowl. I first suspected an
artifact of its low-texture specular surface; the frames say otherwise. Under the standing
framing -- any parameter that produces a PLAUSIBLE physical motion will do -- this fails
outright: no density, friction or restitution turns a pot into a bowl. It is not
identifiability and not frame rate, it is asset integrity, and it is a different problem
from the one being solved.

**A mistake made twice.** The first appearance-based audit used PEAK displacement and
scored 38 px of "motion" for a vase that never leaves its spot, because it transiently
splits into two blobs and re-forms. End-to-end displacement in object-widths moved six
takes from MOVES to STATIC. This is exactly the M39 path-length lesson -- a statistic
that accumulates transients is not a displacement -- repeated in the replacement.

**What still stands from M39.** The estimator comparison: the retired grid fitter
reported friction 0.680 "established" where the ten individual takes span 0.007-0.285.
That ran both estimators over IDENTICAL tracks, so no tracking problem affects it.

**What is now weaker.** The 0-of-15 parameter result is computed from the same point
tracks the audit found unreliable. It should be read as "not established", not as
"proven unmeasurable". Re-deriving parameters on validated tracks is the outstanding work.

**Retired:** the >= 60 fps vendor requirement. Frame rate cannot be the binding
constraint while the instrument is this noisy, and per the standing framing there is no
ground truth to be accurate against -- a plausible parameter that explains the clip is
the bar.

**Known limits.** `rubber_duck_collide_seed0` classes as moving (0.55 object-widths)
where visual inspection says it re-forms rather than translates; NCC cannot fully
separate "translated" from "re-rendered nearby". The audit is calibrated against ~12
clips inspected frame by frame, not all 93.

---

## M41 — parameters re-derived on appearance tracks

M40 left the parameter table resting on the point tracks it had just discredited.
Re-derived from NCC patch tracks (`scripts/rederive.py`), dropping DEGRADED takes since
no rigid-body theta explains an object that stops being itself.

**Subpixel refinement added** to `patch_track`. `cv2.matchTemplate` peaks on the integer
grid, and every parameter here is a velocity fitted over ~5 frames, so +-0.5 px of
quantisation is the same size as the real tracking noise. A parabola through the peak and
its neighbours recovers the fraction. Validated on a synthetic translating square:
recovered 2.380 px/frame against a true 2.400, per-frame scatter 0.29 px.

**The noise floor is now measured, not assumed.** The takes in which nothing moves contain
only noise by construction, so their residual scatter calibrates every other clip:
**2.04 px**, against the 1.50 px previously assumed. Every earlier error bar was
understated by about a third.

**Coverage 6 -> 11 of 15 parameters.** The rubber duck yielded nothing at all before and
now yields all three.

    object        probe     point tracks (published)   appearance tracks
    baseball      drop      0.000 +- 0.039  (n=3)      0.215 +- 0.060  (n=2)
    baseball      slide     0.014 +- 0.032  (n=10)     0.009 +- 0.047  (n=10)
    apple         slide     0.014 +- 0.004  (n=2)      0.004 +- 0.002  (n=1)
    brass_pot     drop      0.050 +- 0.084  (n=2)      0.166 +- 0.202  (n=1)
    brass_pot     collide   0.508 +- 0.472  (n=2)      1.441 +- 0.508  (n=1)
    ceramic_vase  slide     0.007 +- 0.000  (n=1)      0.039 +- 0.022  (n=3)
    ceramic_vase  drop      —                          0.123 +- 0.039  (n=2)
    rubber_duck   drop      —                          0.116 +- 0.099  (n=2)
    rubber_duck   slide     —                          0.080 +- 0.036  (n=2)
    rubber_duck   collide   —                          0.676 +- 0.295  (n=1)

**The published baseball restitution was 0.000 -- no bounce at all -- against 0.215.**
The point tracker was missing the rebound outright, so that row was WRONG, not merely
imprecise.

**Precision did not improve: 0 of 15 are tighter than +-25% from more than one take.**
Closest are baseball restitution (+-28%) and vase restitution (+-32%).

**Where the takes go now:** 14 degraded, 3 contradicted physics, and **50 classified as
moving but yielding no extractable observable** -- the object moves, but the specific
event (a landing, a departure after impact) is not recoverable. That is the dominant loss
and is uncharacterised.

**Least trustworthy rows.** The rubber duck's three values come from exactly the clips
flagged in M40 as re-forming rather than translating. The brass pot's two are single takes
after 14 of its clips were dropped.

Report regenerated; the comparison figure shows both tracking layers side by side.

---

## M42 — where the wasted takes go, and the expanded lab

**Diagnosed the 50 "moves but yields nothing" takes before spending GPU on more of them.**
Track caching added to `rederive.py` so this class of question is free from now on.

    20x [collide] mover was not approaching
    17x [collide] target never moved
     5x [collide] target did not depart
     4x [drop   ] no fall
     4x [slide  ] never moved

**42 of the 50 are collisions.** Yield by probe: collide ~24%, drop and slide ~80%. So
generating more of the same collide clips mostly buys more failures, and the same GPU
hours buy roughly three times the data on the probes that work.

**Expanded lab** (`blender_expand.py`, `gen_expand.py`): all seven scene objects -- the
wooden bowl and the book have never been probed -- at three drop heights plus a slide.
28 experiments x 6 seeds = 168 clips. Collide is deliberately absent.

**New parameter: e(v).** Restitution is not one number; for real materials it falls as
impact speed rises. Three heights (0.08 / 0.18 / 0.30 m, spanning ~1.25 to ~2.4 m/s)
turn the best-yielding probe into a measurement of

    e0     restitution extrapolated to zero impact speed
    de/dv  its change per m/s of impact speed

and simultaneously triple the samples behind each object's e. The impact speed is
MEASURED from the track, not assumed from the staged height -- the video model does not
promise to obey g, so a height that produced no visible fall contributes nothing rather
than contributing a wrong x-value.

Fitting is a weighted straight line through (v_impact, e) with each take weighted by its
own error bar; the slope's interval says whether these clips resolve any speed dependence
at all.

---

## M43 — the expansion pays off: three parameters clear the precision bar

168 clips, 7 objects, 3 drop heights + slide. **Three parameters now measured to better
than +-25% from more than one take, against zero before.**

    ceramic_vase restitution   0.094 +- 0.014   (15%)
    rubber_duck  restitution   0.125 +- 0.027   (22%)
    wooden_bowl  friction      0.317 +- 0.076   (24%)

The wooden bowl's friction is also the first value that lands in the textbook range for
wood on wood (0.3-0.6) rather than an order of magnitude below it. That is consistent
with the M39 rolling-vs-sliding finding: the bowl is flat-bottomed and actually SLIDES,
whereas the sphere-shaped objects roll, and a rolling object's deceleration measures
rolling resistance rather than the Coulomb mu the simulator consumes. Probe validity
depends on object shape, and the bowl is the first object for which the slide probe is
the right experiment.

Full table (e at the centre of the measured speed range; de/dv per m/s):

    object          n     e at mid speed          de/dv        friction mu     v_mid
    apple           7    0.004 +- 0.005   -0.012 +- 0.008   0.026 +- 0.012   2.85 m/s
    baseball        6    0.082 +- 0.023   +0.517 +- 0.080   0.001 +- 0.024   1.09 m/s
    book            0            —                —          0.017 +- 0.182
    brass_pot       2            —                —          0.079 +- 0.027
    ceramic_vase   12    0.094 +- 0.014   +0.034 +- 0.038   0.008 +- 0.026   0.98 m/s
    rubber_duck     4    0.125 +- 0.027   -0.849 +- 0.218   0.029 +- 0.059   0.66 m/s
    wooden_bowl     6    0.057 +- 0.044   +0.188 +- 0.429   0.317 +- 0.076   0.51 m/s

**MY YIELD ESTIMATE WAS WRONG.** M42 justified dropping collide by claiming drop and
slide yield ~80%. Actual drop yield here is **29%** (low 33%, mid 36%, high 19%). The
error: I read "MOVES" from the M40 audit as "yields a usable observable", and they are
different questions -- an object can move without the specific event being extractable.
Dropping collide was still right (24% and falling), but for a weaker reason than stated.

**"No fall" is the dominant failure at every height** -- 60 of ~89 drop failures. The
object does not visibly fall even when staged 30 cm in the air. This qualifies the M40
screen conclusion that airborne objects do fall: they do, in roughly half of clips.

**Height matters, and higher is worse:** 19% at 0.30 m against 36% at 0.18 m, with 12
degraded takes at the high setting against 4 at mid. Staging an object further from its
rendered context makes the model more likely to re-render it rather than move it.

**The book is a failed probe object: 0 of 18 drops usable** (7 no fall, 6 degraded, 5
with e > 1, up to 3.70). A flat slab tumbles when dropped, and a tumbling object lifts
its own tracked centroid, which reads as a rebound larger than the fall. The book needs a
different probe or an orientation-aware tracker.

**A prediction I said I would check, and could not.** I predicted the baseball's
physically-backwards de/dv = +0.517 would flip sign or lose significance with the full
data. It did neither -- but only because the baseball's drop clips were already complete
at the 45-clip partial run, so the fit is over the identical 6 takes. The prediction was
never actually tested. Of the two slopes that are resolved, the duck's -0.849 has the
physically expected sign (restitution falls with impact speed) and the baseball's does
not.

---

## M44 — generated video against our simulation of it

The acceptance test the project has been circling: run Newton at the parameter recovered
from a clip, and put the two side by side (`make_sim_videos.py`, `make_comparison_figs.py`).

There is no photoreal renderer here, so the simulated pane is a composite -- the object is
inpainted out of the staged frame with `cv2.inpaint` and its own sprite pasted wherever the
rollout puts it. Every pixel of the object is real; every position is simulated. Stated
plainly in the report rather than presented as a render.

    ceramic_vase  drop   e measured 0.094, simulated 0.094   MATCHES
    rubber_duck   drop   e measured 0.125, simulated 0.194   NOT REALISABLE
    wooden_bowl   slide  mu 0.317, launched at 0.79 m/s from the clip

**The vase matches exactly.** Under the standing framing -- any plausible parameter that
explains the video will do -- that is the whole acceptance test, and it passes.

**The duck does not, and the failure is informative.** No contact damping between 0.5 and
2000 reproduces e = 0.125 for that mesh; the simulator floors at 0.194. The recovered
parameter is not physically realisable in our simulator. That is a genuine negative
result rather than a fitting failure, and precisely the check that treating the simulator
as a hard constraint exists to perform.

**Three bugs found by looking at the output rather than trusting it:**

  * `rollout()` always started the body at zero velocity, so the simulated bowl never
    slid at all -- the comparison would have shown a stationary object against a moving
    one and looked like a physics disagreement. Launch speed is part of the observation,
    not the material, so it is now measured from the clip (0.79 m/s).
  * Clip selection picked the best-tracked take rather than one that actually PRODUCED a
    measurement, so the chosen slide clip had no usable observable and the sim was
    launched at 0 m/s. Selection now requires the take to yield an observable.
  * The bisection returned its LAST iterate rather than its closest. When the target lies
    outside what the simulator can produce -- exactly the duck's case -- the last iterate
    is an arbitrary endpoint. Keeping the closest is what turned "cd=62, e=0.308" into
    the honest "the floor is 0.194".

Also: GIF at full frame rate came to 4.3 MB *each*. Animated WebP at 12 fps carries the
same 49-frame comparison in ~90 KB, which is what makes it embeddable.

Report rebuilt around the expanded lab: the three parameters that clear +-25%, the
side-by-side animations, the seven-object e(v) table, drop yield by height, and the full
corrections list.

---

## M45 — the tracker was aimed by the wrong pipeline; three results withdrawn

Chasing a visual defect the user reported in the simulated pane ("the vase is floating mid
air and some shadow drops") led to a foundational bug, and the M43/M44 headline results do
not survive it.

**The seeds were off the object.** `seeds.json` came from projecting `obj_geom`'s body
centre, computed as `pos + vmean` -- the mesh's vertex mean added to its placed position,
with `rot_z` never applied. Blender places the origin at `pos` and rotates it. The two
conventions differ by the mesh's own centroid offset: ~0 for a baseball, large for
anything asymmetric.

    object          seed u,v    true u,v      offset   patch half-size
    ceramic vase   186, 122    151, 114     +35,  +8        28
    rubber duck    183,  98    147, 118     +36, -20        29
    brass pot      195, 108    155, 196     +40, -88        41
    book           208, 135    181, 138     +27,  -3        16
    baseball       154, 135    152, 137      +1,  -2        17

Where the offset exceeds the patch half-size the tracker was following mostly wall. This
is why the inpaint mask missed the vase and left it hanging while a fragment fell: the
mask was centred 35 px to the side of it.

**The fix removes the parallel pipeline rather than correcting it** (`seed_from_image.py`).
The subject is the only thing that moves between two stagings of the same object, so
differencing two initial frames isolates it exactly, in the image we are about to track.
Geometry conventions cannot disagree with the renderer if the renderer is the source.
18 of 28 seeds moved by more than 20 px.

**WITHDRAWN, on corrected seeds:**

    wooden_bowl friction     0.317 +- 0.076  ->  0.062 +- 0.151   not a measurement
    rubber_duck restitution  0.125 +- 0.027  ->  0.025 +- 0.010   (+-40%)
    ceramic_vase restitution 0.094 +- 0.014  ->  0.033 +- 0.008

The wooden bowl's friction was reported as "the first value to land in the textbook range
for wood on wood", with a tidy explanation (flat-bottomed objects slide, spheres roll).
The explanation was plausible and the number was an artefact. Both surviving parameters
now belong to the same object:

    ceramic_vase restitution  0.033 +- 0.008   (24%)
    ceramic_vase friction     0.026 +- 0.006   (22%)

**The generated video runs ~5x too slow.** Separately measured while diagnosing why the
simulated pane looks inert: across 90 usable drops the generated fall takes a median 5.5x
longer than physics allows (quartiles 3.2-7.8x), an effective gravity near 0.3 m/s^2. Our
rollout finishes falling, bouncing and settling in the first quarter of the clip because
a 0.18 m fall really does take 4.6 frames at 24 fps.

This splits the parameters by dimension. Restitution is a RATIO of two speeds in the same
clip, so a uniform time rescale cancels and it survives. Friction is |a|/g, dimensional:
stretch time by alpha and measured acceleration falls by alpha^2. Every friction number is
therefore suspect in a way the restitutions are not. No correction is applied because we
cannot yet show the same stretch applies to horizontal motion -- a drop carries its own
clock (known height, known g), so staging a drop and a slide in one clip would let one
calibrate the other.

**Three further bugs, all found by looking rather than trusting:**

  * the sprite and inpaint mask were sized from `size_px`, the SMALLEST horizontal extent
    -- right for seeding a tracker, wrong for cutting out a 12.6 x 24.8 cm vase, so the
    mask covered 44 px of a 120 px object;
  * the cd bisection aborted on any rollout with no detectable rebound, returning
    cd=32/e=0.001 for a target of 0.033 while cd=4 gave 0.094 -- the answer was bracketed
    the whole time and simply never found. A rebound-free rollout is e~0, not a reason to
    stop searching;
  * `drop_observables` divided by a `v_post` of exactly zero.

**Still standing:** the ceramic vase's drop is reproduced by a Newton rollout (measured
0.033, simulated 0.032) and the rubber duck's is not (measured 0.025, simulator floors at
0.261). Worth stating plainly: the vase also "matched" at the old, wrong value of 0.094.
A matching simulation confirms a parameter is REALISABLE, not that it is RIGHT.

---

## M46 — the simulator was holding the objects on their sides

Chasing the user's report that the simulated GIF was "all mixed up" ended in a question
that cut through everything: *why do you even have masks?*

**The masks existed only because the simulated video was faked.** The object was cut out
of a photograph with a mask and pasted wherever the rollout put it. Nothing in the physics
or the measurement needs a mask, and every visual defect was one failing to match the
object's shape -- an ellipse too small for a tall vase (top and bottom left floating, the
mid-band falling as a "shadow"), an ellipse against a bulb-with-neck (neck faded out, smear
left behind), two overlapping instances cut out together (dark blob, shadow trapezoid).
Four rounds of fixes, each revealing the next artefact: the signature of a wrong approach.

**Rendering the actual simulated mesh exposed the real bug in one frame.** The vase came
out lying on its side.

    glTF is Y-up by specification. Blender's importer converts to Z-up on load;
    trimesh does not, and load_asset never applied the rotation.

    object          simulated footprint   simulated height   actual
    ceramic vase       12.6 x 26.5 cm         13.1 cm        12.6 x 12.6, 24.8 tall
    wooden bowl        19.3 x  5.7 cm         19.1 cm        19.4 x 19.2,  5.8 tall
    rubber duck        13.3 x 17.5 cm         18.4 cm        correct by luck

The vase had been falling on its side and the bowl standing on its rim for the entire
project. Sphere covers, resting heights and all contact geometry were built from the wrong
pose, so **every simulation-side result predating this fix is void** -- including M44's
"the vase's drop is reproduced" and the duck's "not realisable" floor of 0.194.

Measurements are unaffected: restitution and friction are read from tracked pixel motion
and never touch the mesh. The vase's e = 0.033 +- 0.008 and mu = 0.026 +- 0.006 stand.

**The compositing hack did not merely look wrong -- it CONCEALED this bug.** The fake pane
pasted a photograph of an upright vase, so the simulator's sideways one was never visible.
The user pushed on a cosmetic-looking defect three times and it turned out to be
load-bearing.

**After the fix** (`load_asset(..., up_convert=True)`, 6 of 7 objects now match their scene
dimensions; the vase is 26.5 cm against a recorded 24.8, a 7% size difference that is
separate from the orientation bug):

    ceramic_vase drop   e measured 0.033, simulated 0.036   rebound agrees
    rubber_duck  drop   e measured 0.025, simulated 0.000   still not reproduced

The vase render now starts upright and TOPPLES on landing while the generated clip keeps it
standing. The rebound magnitudes agree; the post-impact behaviour does not. A tall narrow
vase toppling from 18 cm is not obviously wrong, and Cosmos holding it perfectly upright is
not obviously right -- this is the first comparison in the project where that question is
legible at all.

**Deleted:** `make_sim_videos.py` and `make_overlay_videos.py`, the entire masking and
compositing path. Comparisons are now whole generated frames beside whole rendered frames
(`src/render/mesh_raster.py`, `export_sim_poses.py`, `render_sim_raster.py`). The renders
are flat-shaded rather than photoreal, which is the honest trade for having nothing
fabricated on screen.

Newton's own viewers (ViewerGL/ViewerUSD) cannot consume this state: ProbeScene is pure
Warp with custom kernels and never builds a newton.Model. warp.render.OpenGLRenderer needs
pyglet, which is not installed, so the mesh is rasterised directly through our own camera.

---

## M47 — render with Newton's own viewer, not a hand-written one

The conditioning frames the video model sees come from **Blender** (`blender_expand.py`,
Cycles, HDRI, the real table asset). I had claimed "there is no photoreal renderer here",
which was simply false -- a photoreal renderer produced every input to the pipeline.

**Why Newton's viewers had nothing to draw.** Newton IS used in this project
(`src/sim/scene_sim.py` builds a `newton.ModelBuilder` with `SolverXPBD` and
`CollisionPipeline`). But `ProbeScene`, the simulator every measurement comes from, is
hand-written Warp kernels and never constructs a `newton.Model`. So it was never "Newton
cannot render this" -- it was "we are not using Newton for this simulation."

That is a reason to give Newton a Model, not to write a third renderer. The geometry is now
loaded into a `ModelBuilder` purely for display and the body transform driven frame by frame
from our own rollout, then rendered with `newton.viewer.ViewerRTX(headless=True)` and
`save_screenshot`. Needed two packages that were missing:

    pip install ovrtx      (ViewerRTX backend, ~1.8 GB wheel)
    pip install pyglet     (>=2.0)

**Deleted:** `src/render/mesh_raster.py` and `scripts/render_sim_raster.py`, the
hand-written rasteriser. It was the wrong answer to the question "how do I see the
simulation" when a real renderer was already in the loop, and a second wrong answer after
the mask compositing.

Remaining mismatch, stated rather than hidden: ViewerRTX draws its own ground and lighting
rather than the wooden table and HDRI the clips were staged in, and its `set_camera` takes
pitch/yaw but not the lab camera's 46 degree field of view. So the panes differ in framing
and surroundings; only the motion is comparable. Rendering the rollout in Blender would put
both panes in the same visual space -- `scripts/blender_render_sim.py` already does this and
is the obvious next step.

---

## M48 — the stack, stated plainly: physics and rendering meet only at a transform

Comparisons now use the Blender/Cycles renders of the rollout, in the staged scene: same
table, same HDRI, same camera, same materials as the clip. The only difference between the
two panes is the physics, which is what makes the comparison legible for the first time.

    layer          holds                                        knows nothing about
    Warp kernels   802 spheres (vase), poses over time          texture, lighting, camera
    Newton         Model/State, solvers, collision pipeline     our custom 6-DOF integrator
    Blender        textured glTF, HDRI, table asset, camera     contact, gradients

They exchange exactly one thing: per-frame object transforms. `export_sim_poses.py` writes
them, `blender_render_sim.py` drives the scene with them. Nothing else needs to be shared,
which is why the rollout can be rendered by whichever renderer already has the scene -- and
the one that already had it was Blender, since it produced every conditioning frame.

Newton's ViewerRTX renders the same rollout (kept, `nsim_*/`) but draws its own ground and
lighting and exposes pitch/yaw without a field-of-view setting. That makes it a
solver-debugging view, not something comparable to a photograph. It is the right tool for
"is the solver doing something sane", the wrong one for "does this match the video".

Worth recording: the physics proxy and the rendered mesh are different representations of
the same object -- 802 spheres versus a textured glTF -- connected only by a transform.
That is normal practice, and it is exactly where the Y-up/Z-up bug hid for the whole
project: nothing ever checked that the two representations agreed.

---

## M49 — the gravity claim was an artefact; a control finally exists

**M45's "the generated video runs a median 5.5x slower than physics, effective gravity
~0.3 m/s^2" is WITHDRAWN.** It came from a measurement that fails on a control where the
answer is known.

The control is new and only became possible once the rollout could be rendered: our own
simulation, drawn by Cycles, where gravity IS 9.81 by construction. Running the same
measurement on it returned **-0.31 m/s^2**.

Decomposing where the error lives:

    measured from                 window                        g
    world z, no camera            6 samples (true free fall)    7.10 m/s^2
    image y, through the camera   6 samples                     9.44 m/s^2
    either                        7 samples (1 past contact)    sign flips

**With the correct window the pipeline recovers 9.44 against a true 9.81 -- 4% -- through
the camera and the pixel scale.** The physics, the projection and the px<->m conversion are
all sound. The fault was entirely window selection:

  * the original method took `argmax(y)`, the lowest point of the trajectory. For a vase
    that lands, rebounds and then topples, that is well after impact, so the parabola was
    fitted across all three phases;
  * the first correction used a deceleration threshold and picked 4 frames instead of 5 on
    the control, giving 13.6 m/s^2 (+39%).

A 0.18 m fall is five frames at 24 fps. One frame either side of the true contact inverts
the answer, which is why every per-clip figure the old method produced (7% to 88% of g
across seeds) is void.

**Current honest status: the generated video's gravity is UNMEASURED, not wrong.** Nothing
should be claimed about it until the free-fall window is selected robustly -- most likely by
detecting contact from the trajectory's curvature rather than a threshold, and by using the
0.30 m drops, which give ~6 frames instead of ~5.

The wider lesson is the value of the control. For the whole project the simulation could
only be inspected through numbers; once it could be rendered, a measurement with a known
answer became available and immediately falsified a published claim. Any future observable
should be run against the rendered simulation before it is run against video.

---

## M50 — ViewerRTX renders the staged scene; the stack, stated

The earlier ViewerRTX pass looked bare -- untextured mesh on a default ground -- and that
read as a Newton limitation. It was not. `log_mesh` takes UVs, a texture, roughness and
metallic, so the staged scene rebuilds inside the viewer: table, duck, apple, book, bowl,
baseball and brass pot, all textured, ray traced, with the simulated body's vertices carried
to the rollout's pose each frame. Entirely Warp/Newton, no Blender in the loop
(`render_sim_rtx_scene.py`).

**How the pieces fit.**

    SIMULATION            src/sim/probe_scene.py, pure Warp kernels
                          802 spheres approximating the vase; 6-DOF integration;
                          penalty contact against a ground plane; differentiable
                          -> per-frame position + quaternion

    INTERFACE             sim_poses.json: one transform per object per frame
                          this is the ONLY thing the two halves exchange

    RENDERING             newton.viewer.ViewerRTX  (Warp-native, path traced)
                          or Blender/Cycles        (what produced the conditioning frames)
                          textured glTF, camera, lights -- knows nothing about contact

Newton the *framework* is used elsewhere (`scene_sim.py`: ModelBuilder, SolverXPBD,
CollisionPipeline) but not by `ProbeScene`, which was hand-written in Warp because Newton's
XPBD/VBD solvers returned zero gradients through contact. So the physics is Warp; Newton
contributes the renderer here.

Remaining gap between the panes: ViewerRTX offers only 'default' / 'studio' / 'none'
environments, so the HDRI street backdrop the clips were staged against is absent, and
`set_camera` takes pitch/yaw with no field-of-view control, so framing differs slightly.
Blender matches both exactly and remains available via `run_sim_and_render.py`.

---

## M51 — the HDRI cannot be matched inside ViewerRTX; simulation and rendering are fully separable

**Attempted and failed:** giving ViewerRTX the same street HDRI the clips were staged
against. It exposes only 'default' / 'studio' / 'none' lighting presets, and
`add_background_usd()` adds background GEOMETRY -- its docstring cites Gaussian splat scans
-- not environment lighting. Authoring a USD `UsdLux.DomeLight` carrying the .hdr and
referencing it rendered **pure black** under `environment='none'`: the dome was ignored and
the presets were disabled at the same time. Reverted to 'studio', the closest available
match. Blender reproduces the HDRI exactly and remains available via
`run_sim_and_render.py`; `make_hdri_dome.py` is kept as the record of the attempt.

Also fixed: static geometry now carries its texture only on the first frame. ViewerRTX
stages textures through /tmp, and re-supplying them every frame made it rewrite and reread
the same files 49 times, racing into "Corrupt PNG" on the table.

**Simulation and rendering are fully separable, and the consequence is larger than the
rendering question.** The simulator produces a transform per object per frame for as many
frames as we ask; the renderer draws whichever frame it is handed. Frame 0 is not special
to the simulator at all -- it was special only to Cosmos, which received it as conditioning
and invented everything after it.

So the pipeline can produce a complete physically-generated video of its own:

    video model    one staged frame -> 49 invented frames   (observation, used to infer)
    our pipeline   recovered theta  -> 49 simulated frames  (physically valid by construction)

That is the project's stated goal rather than a side effect: recover theta, then the
simulator produces editable, physically-valid animation. The video model is a source of
motion to measure, not the thing that makes the final motion.

---

## M52 — prompting A/B: adopt the vendor negative prompt, on weak evidence, and say so

The Cosmos3-Nano card states that "text prompts should be upsampled into a specific JSON
structure" and ships `assets/example_i2v_prompt.json` (18 keys, ~8k chars) plus
`assets/negative_prompt.json`. This project had been sending a one-key `{"scene": "..."}`
stub and **no negative prompt at all**, while the Hunyuan backend got one. Four arms, two
objects, six seeds each, 48 clips; the staged frame is byte-identical across arms.

    arm                          brass pot          ceramic vase      yield    degraded
    stub (shipped)          0/6, 2 degraded                  2/6    2/12 17%      17%
    stub + negative         3/6, 0 degraded                  2/6    5/12 42%       0%
    JSON structure          2/6, 0 degraded                  2/6    4/12 33%       0%
    JSON + negative         2/6, 0 degraded                  0/6    2/12 17%       0%

**Nothing reaches significance on yield.** Fisher exact vs control: neg p=0.37,
json p=0.64, json_neg p=1.00. The +25 points for the negative prompt is three clips out of
twelve, and the pattern is NON-MONOTONIC -- combining two supposedly helpful interventions
lands exactly on baseline, which is the signature of noise.

**The test was underpowered and I should have checked before spending the clips:** at 12
per arm the power to detect 17% vs 42% is **26%**. Sixty per arm would give 88%.

The one result with any support is degradation: **2/12 with no negative prompt, 0/36 across
every arm that had one, Fisher p=0.06.** It also has a mechanism -- the shipped negative
prompt explicitly names "floating or improperly grounded" and "distorted features", which
is close to a description of the brass pot turning into a bowl. Both degraded clips are
still brass pot, the object that degrades anyway, so this is suggestive rather than shown.

**Adopted:** the vendor negative prompt is now the default for CosmosI2V (vendored to
`assets/cosmos/negative_prompt.json` so it does not depend on a cache path). It is one
line, it is what NVIDIA ships and their own i2v example passes, no arm was worse with it,
and the degradation signal points the right way. That is the argument -- a cheap
vendor-recommended default, not a demonstrated win.

**Not adopted:** the JSON structure. 3000 chars of prompt for no measurable gain and ~30%
slower generation (117s vs 90s per clip).

**A figure-selection bug worth recording.** The first comparison picked the highest-NCC
clip per arm, which selects for STATIC clips -- a motionless object matches its own
template perfectly. It made the negative-prompt arm look like a pot frozen in mid-air.
Re-selecting on "passed the usability test" shows the real behaviour: control degrades into
a white bowl by frame 24, while the other three arms keep a lidded copper pot that falls and
settles. Any figure that picks exemplars by a quality metric has to use a metric that a
failure cannot maximise.

---

## M53 — the contact model was creating energy; "everything is a bit bouncy" was right

The user observed that the simulated objects look too bouncy. That conflicts with our own
numbers -- we measure the vase at e=0.033, nearly dead -- so it was worth testing rather
than tuning. Dropping a single sphere (no mesh, no toppling, no camera, world z only) and
sweeping the contact damping:

    cd      rebound/drop    e
     0.5       0.90        0.949
     5.0       0.42        0.649
    20.0       0.04        0.200
    60.0       0.00        0.000
   200.0       0.00        0.000
   600.0      26.80        5.18     <- ENERGY CREATED
  2000.0      98.39        9.92     <- ENERGY CREATED

**Beyond cd~200 the explicit penalty contact pumps energy**: the ball leaves the ground 98x
higher than it was dropped from, which no passive contact can do. The bound is closed-form
-- the damping impulse must not reverse the velocity within one step:

    cd < 2m/dt = 2(0.127)/0.000694 = 367     observed stable at 200, exploding by 600

**And the bounciness has a number.** Critical damping here is 2*sqrt(km) = 35.7, so the
cd=5 our fit chose is **0.14x critical** -- barely damped, hence visibly bouncy. The fit
chose it because `tune_cd` matches the PROJECTED rebound of a toppling vase, which is not
the physical restitution; the objective was matching the wrong quantity.

**Fixes:**
  * `ProbeScene` now computes `2m/dt` from the actual masses and clamps cd to half of it,
    with a RuntimeWarning. Half, not 0.9x: at 0.9x (cd=330) the model still gave e=0.64
    where cd=200 gives e=0.00. Restitution is monotonic in cd only well below the bound.
  * the cd search in `export_sim_poses.py` is bounded to [0.5, 200], the stable monotonic
    region. It had been [0.5, 2000], i.e. half the search space was an artefact -- so any
    cd it returned above ~200 was meaningless, including in the duck's "floor at 0.194".
  * verified after the fix: e is monotonic non-increasing over cd in [0.5, 160], spanning
    0.949 down to 0.000, and cd=600/2000 now clamp to a physical answer.

This is the third time a control with a known answer has falsified something (after the
sideways meshes and the gravity claim). A single sphere has an analytic rebound; nothing
about this needed video, tracking, or a mesh.

---

## M54 — the objective could not measure restitution at all

The contact fix (M53) made the simulator honest but left the thing being matched wrong.
Validated against a control where truth is known -- a clean sphere, no mesh, no toppling,
restitution read from world z -- the old objective was not merely biased:

    cd     true e    OLD objective    NEW objective
     1      1.079        0.439           1.057
     2      0.983        0.302           0.961
     5      0.746        0.000           0.728
    10      0.477        0.000           0.464
    20      0.188        0.000           0.182
    40      0.037        0.000           0.035
                      MAE 0.44        MAE 0.014, monotonic

**It returned 0.000 for every cd >= 5.** It could not distinguish cd=5 from cd=40, so the
fit was unconstrained over most of the search range -- which is exactly how it settled on
cd=5 (0.14x critical damping, visibly bouncy) while reporting e=0.036.

Two bugs, the same shape as the gravity error:

  * `hit = argmax(y)` took the trajectory's LOWEST POINT. For a vase that lands, rebounds
    and topples, that is long after impact, so the velocities either side straddled the
    topple rather than the bounce. Now `first_contact()` takes the frame of PEAK DOWNWARD
    SPEED, which is first contact by definition and needs no threshold.
  * `WIN=5` least-squares fits averaged a one-to-three frame bounce together with the fall
    that follows, netting out downward. Now the observable is peak speed in, peak speed
    out.

**A smoothing trap, rejected on purpose.** Smoothing the trace suppresses physically
impossible e>1 readings (35 -> 19 of ~145 real drops) and looks like an improvement. It is
not: it also drags the median restitution of every good measurement from 0.300 to 0.042,
a factor of seven, and takes the control's MAE from 0.014 to 0.162. A cleaner distribution
bought by attenuating the signal is not cleaner. Bad takes are rejected by the
admissibility check; good ones are left alone.

**Measurement-quality number that falls out:** 24% of real drops return e > 1 and are
rejected. That is the honest cost of reading restitution from a 24 fps projected centroid,
and no amount of code fixes it.

Expect the recovered restitutions to move up substantially from 0.033 -- that value came
from an objective that could not measure restitution.

---

## M55 — the speed axis, and a false breakthrough caught before publication

The re-fit first reported **7-8 parameters at +-1-7%**, up from 2. That was an artefact and
is the closest this project came to publishing a fabricated result.

**Bug 1: the covariance ignored between-take disagreement.** `fit_expand`'s weighted line
fit built its covariance from the per-take error bars alone. The apple's twelve takes span
e = 0.000 to 0.998 (std 0.371) while their individual bars are ~0.05 -- a 40x mismatch --
and the fit reported +-0.009. Friction escaped it only because `combine()` carries a
between-take term, which is why friction read an honest +-22% while every restitution read
+-1-7%. All seven "new" parameters came from the broken path. Fixed by scaling the
covariance by reduced chi-square, the standard treatment when residuals exceed the bars.

**Bug 2: a horizontal pixel scale used for vertical motion.** Drops are measured along
image y, but the fit converted to m/s with the scale computed for horizontal motion along
the lane: 345 px/m where the vertical scale at the drop point is 490, a factor of 1.42.
Validated on the control -- with the vertical scale, measured impact speed matches world-z
truth to within 0.07 m/s at all three heights.

**Bug 3: peak-picking latched onto tracker jumps.** Even after the scale fix the reported
impact speeds were ~2x the physical ceiling. An object released from rest cannot arrive
faster than free fall from its staged height, so exceeding sqrt(2gh) is a tracker artefact,
not a fast clip -- the same class as e > 1 and rejected the same way. **40 of 168 takes in
the old lab and 34 of 168 in the new one are impossibly fast.**

**Results after all three fixes, both labs, same estimator:**

    lab                        parameters within +-25%
    expand      (no negative)   1   ceramic_vase friction 0.026 +- 0.006
    expand_neg  (negative)      5   apple e 0.527+-0.092, brass_pot e 0.364+-0.066,
                                    ceramic_vase e 0.266+-0.048, rubber_duck e 0.244+-0.054,
                                    rubber_duck friction 0.039+-0.009

**1 vs 5 at n=168 each is the properly-powered answer** the 12-per-arm A/B could not give
(26% power). The negative prompt earns its place on this evidence, not on the earlier
p=0.37.

**Two caveats kept in view.** The de/dv slopes are mostly the WRONG SIGN (apple +0.400,
pot +0.249, vase +0.151 -- restitution rising with impact speed, which real materials do
not do), so speed dependence is still not credible and should not be reported. And the
count depends on the interval convention: `e_mid` is exactly the plain weighted mean, but
`combine()`'s more conservative between-take interval is wider (apple +-0.131 vs +-0.092),
under which about 3 of the 5 clear +-25% rather than 5.

The restitutions are now physically plausible for the first time -- apple 0.53, ceramic
0.27, duck 0.24 -- where the broken objective had given 0.03 for everything.
