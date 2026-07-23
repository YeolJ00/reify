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
