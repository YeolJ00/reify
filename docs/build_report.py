"""Build docs/report.html — self-contained (inline CSS, base64-embedded plots).

Regenerate after new results: python docs/build_report.py
Keep the narrative in sync with docs/PROJECT_LOG.md (Claude-facing).
"""

import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"


def img_tag(name, caption):
    p = OUT / name
    if not p.exists():
        return f'<p class="missing">[missing plot: {name}]</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


CSS = """
  :root { --fg:#1c2128; --bg:#fcfcfd; --accent:#0b5fa5; --soft:#f1f4f7;
          --line:#d7dee6; --muted:#5b6672; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e6e9ed; --bg:#14181d; --accent:#6ab0e8; --soft:#1d242c;
            --line:#333d48; --muted:#98a4b0; }
  }
  :root[data-theme="dark"] { --fg:#e6e9ed; --bg:#14181d; --accent:#6ab0e8;
    --soft:#1d242c; --line:#333d48; --muted:#98a4b0; }
  :root[data-theme="light"] { --fg:#1c2128; --bg:#fcfcfd; --accent:#0b5fa5;
    --soft:#f1f4f7; --line:#d7dee6; --muted:#5b6672; }
  body { font: 16px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif;
         color:var(--fg); background:var(--bg);
         max-width: 900px; margin: 0 auto; padding: 2rem 1.2rem 4rem; }
  h1 { font-size: 1.7rem; line-height:1.25; letter-spacing:-.015em;
       border-bottom: 2px solid var(--accent); padding-bottom:.4rem;
       text-wrap: balance; }
  h1 + p em { color: var(--muted); font-style: normal; font-size:.9em; }
  h2 { font-size: 1.25rem; color: var(--accent); margin-top: 2.4rem;
       letter-spacing:-.01em; text-wrap: balance; }
  h3 { font-size: 1.02rem; margin-top: 1.6rem; }
  code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace;
              background: var(--soft); border-radius: 4px; }
  code { padding: .1em .35em; font-size: .86em; }
  pre { padding: .8rem 1rem; overflow-x: auto; font-size: .82em; line-height: 1.5; }
  table { border-collapse: collapse; margin: 1rem 0; font-size: .9em;
          font-variant-numeric: tabular-nums; display:block; overflow-x:auto; }
  th, td { border: 1px solid var(--line); padding: .35rem .65rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: var(--soft); font-weight:600; }
  figure { margin: 1.4rem 0; text-align: center; }
  figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }
  figcaption { font-size: .84em; color: var(--muted); margin-top: .45rem; }
  .finding { background: var(--soft); border-left: 4px solid var(--accent);
             padding: .7rem 1rem; margin: 1.1rem 0; border-radius: 0 6px 6px 0; }
  .missing { color: #c0392b; }
  .tag { display:inline-block; background:var(--accent); color:var(--bg);
         border-radius: 10px; padding: 0 .55em; font-size:.72em; font-weight:650;
         letter-spacing:.03em; vertical-align: middle; margin-right:.45em; }
"""

HEAD = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inverse Simulation from Video Priors — M0–M4 Report</title>
<style>{CSS}</style></head><body>
"""


def build(m4_result_html: str, artifact: bool = False) -> str:
    if artifact:
        # Artifact pages get the html/head/body skeleton from the publisher;
        # provide only <title>, <style>, and content.
        s = ("<title>Inverse Simulation from Video Priors — M0–M4 Report</title>\n"
             f"<style>{CSS}</style>\n")
    else:
        s = HEAD
    s += """
<h1>Inverse Simulation from Video Priors — M0–M4 Report</h1>
<p><em>2026-07-21 · Newton 1.4.0 / Warp 1.15.0 · RTX A6000 · repo:
<code>simulation-assestization</code></em></p>

<p>Goal: given a roughly-initialized simulation asset (geometry and placement fixed)
and a motion observation, recover the physics parameters <b>θ</b> whose Newton
simulation reproduces the <em>physically realizable</em> part of the motion. The
simulator is a hard constraint; we optimize per instance, no trained model.
Test scene: a cloth flag (1.2×0.8&nbsp;m, 425 particles) pinned along its hoist edge,
driven by gravity and a low-dimensional wind forcing.</p>

<h2><span class="tag">M0</span> One differentiable rollout on GPU</h2>
<p>Newton has no built-in wind, so we apply an aerodynamic normal-pressure force per
triangle, <code>f = c·A·(n̂·(v_wind − v_cloth))·n̂</code>, written into
<code>state.particle_f</code> before each substep by a Warp kernel — fully
differentiable. 60 frames × 32 substeps runs in <b>0.40&nbsp;s</b> (semi-implicit
solver; VBD is ~35× slower per substep here). A first attempt exploded: explicit
damping needs <code>kd·dt/m ≪ 1</code> — worth remembering that the stability
envelope is part of the optimization landscape (see the cliff in M3).</p>
"""
    s += img_tag("forward_semi_implicit.png",
                 "M0 — forward rollout: the flag droops from the pinned edge and flaps downwind.")
    s += """
<h2><span class="tag">M1</span> Gradient through the rollout, verified</h2>
<div class="finding"><b>Key finding.</b> Gradients flow through
<code>SolverSemiImplicit</code> and match finite differences to 0.3&nbsp;%. They do
<b>not</b> flow through <code>SolverVBD</code> (tape returns exactly 0.0) — so VBD
work uses zeroth-order optimizers, which our final pipeline supports natively.</div>

<h2><span class="tag">M2</span> Single-parameter recovery</h2>
<p>Recover wind strength (true 15.0) from a synthetic target, starting at 8.0:
Adam on the tape gradient → <b>15.09</b> (0.6&nbsp;%); CEM (with decaying exploration
noise to prevent premature collapse) → <b>14.99</b> (0.05&nbsp;%).</p>

<h2><span class="tag">M3</span> Full θ (8 params) + identifiability</h2>
<p>θ = [wind Fourier a0, a1, a2 · gravity_z · log tri_ke · log tri_kd ·
log edge_ke · log mass]. All flow through the tape via fill-kernels. Findings, each
established by running:</p>

<h3>The landscape, not the gradient, is the problem</h3>
<p>The adjoint verifies near the target, but far away the flapping dynamics make the
loss rugged — finite differences never converge as h shrinks, and plain Adam stalls.
The Gauss-Newton spectrum explains why: condition number <b>3.3×10⁵</b>. Uniform-step
optimizers overshoot stiff directions while starving sloppy ones.</p>

<h3>What is identifiable</h3>
<table>
<tr><th>parameter</th><th>RMS trajectory sensitivity</th><th>verdict</th></tr>
<tr><td>log mass</td><td>0.448</td><td rowspan="2">strong (their ratio is the
stiffest direction)</td></tr>
<tr><td>log tri_kd (damping)</td><td>0.405</td></tr>
<tr><td>log tri_ke (stretch)</td><td>0.083</td><td>good</td></tr>
<tr><td>gravity_z</td><td>0.067</td><td>good</td></tr>
<tr><td>wind a0 / a2 / a1</td><td>0.010 / 0.006 / 0.005</td><td>sloppy (~50× weaker)</td></tr>
<tr><td>log edge_ke (bending)</td><td>0.002</td><td>~unobservable</td></tr>
</table>

<div class="finding"><b>Scale gauge freedom — empirically demonstrated.</b> Gravity
enters as an acceleration while all other forces scale by 1/mass, so only force/mass
ratios are observable. A free LM run matched the target trajectory to loss 3×10⁻⁸ with
mass, stiffness, damping and wind all off by the <em>same</em> factor 1.13 — and
gravity exact (0.02&nbsp;%). Consequence: the method needs a scale anchor (e.g. known
density from the asset) or a posterior over the scale direction. This settles the
M3 point-estimate-vs-posterior question.</div>
"""
    s += img_tag("identifiability.png",
                 "M3 — left: FD Hessian structure. Right: 2D loss slice; the bright plateau below "
                 "the optimum is the numerical stability cliff (lighter mass explodes the sim at fixed dt).")
    s += """
<h3>The optimizer that works: multi-start Levenberg–Marquardt</h3>
<p>LM on an FD trajectory Jacobian (16 rollouts/iteration, solver-agnostic — works
with VBD too) handles the ill-conditioning that defeats Adam. Individual runs are a
<em>basin lottery</em> (GPU atomic reordering alone routes identical runs to different
minima), so we run multiple starts and keep the best final loss. We also verified
run-to-run bitwise determinism is available (<code>wp.config.deterministic</code>) at a
7× rollout cost — off by default, on for verification runs.</p>
<p>Result with mass gauge-fixed, 5 starts (423&nbsp;s): 3/5 starts reach loss
~5×10⁻¹⁰ and the winner recovers <b>every parameter to ≤0.07&nbsp;%</b> — even
near-unobservable bending to 1.3&nbsp;%:</p>
<table>
<tr><th>param</th><th>true</th><th>recovered</th><th>rel err</th></tr>
<tr><td>wind a0</td><td>15.0</td><td>15.0015</td><td>0.01 %</td></tr>
<tr><td>wind a1</td><td>4.0</td><td>4.0008</td><td>0.02 %</td></tr>
<tr><td>wind a2</td><td>−3.0</td><td>−2.9980</td><td>0.07 %</td></tr>
<tr><td>gravity_z</td><td>−9.81</td><td>−9.8113</td><td>0.01 %</td></tr>
<tr><td>tri_ke</td><td>5000</td><td>5000.7</td><td>0.01 %</td></tr>
<tr><td>tri_kd</td><td>10.0</td><td>10.003</td><td>0.03 %</td></tr>
<tr><td>edge_ke</td><td>10.0</td><td>9.869</td><td>1.31 %</td></tr>
<tr><td>mass</td><td>0.05</td><td>0.05</td><td>fixed (gauge)</td></tr>
</table>
"""
    s += img_tag("recover_full_lm_fix.png",
                 "M3 — winning LM start: loss falls 6 orders of magnitude in 11 iterations.")
    s += """
<h2><span class="tag">M4</span> Stage 1: recovery from <em>tracked video</em>, not 3D states</h2>
<p>The target is now observed the way real footage will be: the true rollout is
<b>rendered</b> (painter-sorted rasterization with a static per-triangle texture),
<b>tracked</b> with chained pyramidal Lucas–Kanade (forward–backward gate rejects bad
tracks — the architecture's step-8 rejection), tracks are attached to the cloth by
frame-0 barycentric coordinates, and θ is recovered by multi-start LM on <b>2D pixel
residuals</b> through a differentiable pinhole projection (gradient FD-verified).</p>
"""
    s += m4_result_html
    s += img_tag("m4_stage1.png",
                 "M4 stage 1 — left: rendered frame 0 with detected features; middle: last frame, "
                 "LK tracks (red) vs recovered-θ simulation reprojected (cyan); right: LM loss.")
    s += """
<h2><span class="tag">M4·2</span> Stage 2 groundwork: generated video as the motion prior</h2>
<p>Three i2v candidates are downloaded and wrapped behind one interface
(<code>src/video/i2v.py</code>): <b>Wan 2.2 TI2V-5B</b> (first-frame faithfulness),
<b>HunyuanVideo 1.5 480p-i2v</b> (reputed best cloth/physics motion), and
<b>Cosmos3-Nano</b> (NVIDIA physical-AI omnimodel, via vLLM-Omni). Selection will be a
bake-off scored by our own downstream metric — θ-recovery error on scenes with known
ground truth — since public physics benchmarks measure the wrong thing for us.</p>
<div class="finding"><b>Wan plumbing test passed.</b> Generating from our rendered
initial frame: first-frame fidelity |f0−I0| = 1.1/255, camera fully static, plausible
and diverse flag motion across seeds — and our LK tracker holds 96&nbsp;% of features
through gentle-motion clips (25&nbsp;% on the most violent seed, still above the usable
floor). ~60&nbsp;s per 49-frame video on one A6000.</div>
"""
    s += img_tag("i2v_wan5b_contact_sheet.png",
                 "M4 stage 2 — Wan 2.2 TI2V-5B: conditioning frame I0 (left) and generated frames "
                 "for three seeds. Static camera, faithful first frame, diverse cloth motion.")
    s += """
<h3>Real assets acquired</h3>
<p>Six scanned rigid objects (Google Scanned Objects, CC-BY 4.0: lion figure, teapot,
C-clamp, file sorter, triceratops, shark), two cloth-like scans (bath towel, braided
cushion), and a CC0 table + studio HDRI (PolyHaven) — all metric-scale, loading through
<code>src/data/assets.py</code>, reproducible via <code>scripts/fetch_assets.sh</code>.
A scanned object was drop-tested in Newton (decimation → rigid body with
density-derived inertia → XPBD + collision): it falls and settles correctly. The scans
provide geometry; the physical attributes (density, friction, stiffness…) are exactly
the θ our pipeline recovers — that is the assetization.</p>

<h2><span class="tag">Bake-off</span> Which i2v model produces <em>physically projectable</em> motion?</h2>
<p>The video model only sees a static I0 — it cannot know the true physics — so we
score each model by <b>distance to the physical manifold</b>: after multi-start LM
projects the generated motion onto our Newton model family, how large is the
irreducible 2D residual? (Reference: video of a real simulation fits at 4&nbsp;px,
the tracking noise floor.) Trackability and motion magnitude guard against
degenerate near-static outputs.</p>
<table>
<tr><th>backend</th><th>prompt</th><th>mean fit RMS</th><th>track survival</th><th>motion</th><th>gen. time/clip</th></tr>
<tr><td><b>Wan 2.2 TI2V-5B</b></td><td>default</td><td><b>9.5 px</b></td><td>62 %</td><td>84 px</td><td>~60 s</td></tr>
<tr><td>HunyuanVideo 1.5</td><td>default</td><td>19.8 px</td><td>55 %</td><td>53 px</td><td>~600 s</td></tr>
<tr><td>HunyuanVideo 1.5</td><td>scene-neutral</td><td>27.0 px</td><td>88 %</td><td>35 px</td><td>~600 s</td></tr>
</table>
<div class="finding"><b>Verdict: Wan 2.2 TI2V-5B.</b> Closest to the manifold, zero
scene edits, locked camera, most motion, 10× cheapest. Hunyuan's local cloth wrinkles
look gorgeous, but it <em>re-stages the scene</em> — with the default prompt it
hallucinates a physical flagpole + tripod; with a neutral prompt it still re-centers
and rescales the cloth. LK happily tracks that rigid drift (88&nbsp;% survival!) but a
pinned Newton flag cannot translate, so scene drift lands in the fit residual — the
metric punishes exactly what breaks the pipeline. Cosmos3-Nano: weights + client
ready, unscored (needs ~40&nbsp;GB free VRAM + core vllm on this shared box).</div>
<div class="finding"><b>Bonus finding — the projection audits the video model's
physics.</b> Across all Wan seeds, the implied gravity is −0.6 to −1.8&nbsp;m/s² —
5–15× weaker than Earth. Video models generate "dreamy slow-mo" cloth, and our
inverse fit measures that bias quantitatively (with correspondingly inflated implied
wind and stiffness). The pipeline doesn't just consume the motion prior; it
quantifies how physical it is.</div>
"""
    s += img_tag("i2v_hunyuan_contact_sheet.png",
                 "HunyuanVideo 1.5 with the default prompt: pixel-faithful first frame, then a "
                 "hallucinated flagpole + tripod appears and the flag is reframed — scene re-staging "
                 "that breaks frame-0-anchored geometry.")
    s += """
<h2><span class="tag">M5</span> A real scanned object, its physics recovered from motion</h2>
<p>Everything so far used an authored cloth flag. M5 takes a <b>real 3-D scan</b> — a
porcelain teapot from Google Scanned Objects — drops it with an initial launch and
spin onto a ground plane, and recovers its physical attributes θ = {density, friction,
restitution, initial linear + angular velocity} from the resulting motion. The scan
provides geometry; placement is input; the physics is what we infer. That inference
<em>is</em> the assetization.</p>
"""
    s += img_tag("rigid_drop_render.png",
                 "The real scanned teapot mesh running in Newton's contact solver: it falls, "
                 "strikes the ground, and rolls to rest — orientation and all.")
    s += """
<p>Two findings carry over and one is new:</p>
<ul>
<li><b>Contact kills the gradient again.</b> The XPBD solver's tape gradient through
contact is exactly zero (finite differences are not) — the same wall we hit with the
VBD cloth solver. The solver-agnostic multi-start Levenberg–Marquardt we built back at
M3 handles it unchanged; the architectural bet keeps paying off.</li>
<li><b>Density is a gauge freedom — the rigid echo of the cloth scale-gauge.</b> In
ideal rigid contact against a fixed floor, free-fall and the collision laws are all
mass-independent, so the trajectory should not reveal density at all. It barely does
(≈9&nbsp;mm of drift under a 4× density change), and that sliver is a numerical artifact
of the compliant contact model, not real physics. So density is unrecoverable on
principle — exactly as mass was for the cloth. Friction and restitution are legible only
through their contact events; the launch velocities are written plainly in the
free-flight arc.</li>
<li><b>The optimizer runs four ways at once.</b> Each Levenberg–Marquardt start took its
own GPU, four in parallel — turning held capacity into a finished multi-start recovery
in one wall-clock pass.</li>
</ul>
<div class="finding"><b>What recovers is exactly what the motion reveals.</b> Seven
parallel Levenberg–Marquardt starts reach a best fit of 17.7&nbsp;mm RMS, and the
per-parameter accuracy tracks the identifiability ranking almost monotonically: the
launch velocities — written plainly in the free-flight arc — come back to ~15&nbsp;%,
while the contact-governed parameters (restitution, friction) and the roll-axis spin
are poor, and <b>density is driven to the floor</b> (its sensitivity sits 50× below the
leading velocity). The recovered path hugs the true arc until the first bounce, then
drifts — residual pools precisely in the weakly-observed, contact-driven phase.</div>
<table>
<tr><th>parameter</th><th>true</th><th>recovered</th><th>observability</th></tr>
<tr><td>v0y (launch)</td><td>0.15</td><td>0.17</td><td>highest</td></tr>
<tr><td>v0x (launch)</td><td>0.60</td><td>0.70</td><td>high</td></tr>
<tr><td>w0z (spin)</td><td>1.50</td><td>1.15</td><td>fair</td></tr>
<tr><td>restitution</td><td>0.55</td><td>0.27</td><td>low</td></tr>
<tr><td>friction μ</td><td>0.40</td><td>1.49</td><td>low</td></tr>
<tr><td>w0y (roll spin)</td><td>3.00</td><td>1.44</td><td>very low</td></tr>
<tr><td>density</td><td>800</td><td>≈ 0 (floored)</td><td>gauge — none</td></tr>
</table>
"""
    s += img_tag("rigid_recover.png",
                 "Left: recovered COM (dashed) tracks the true free-flight arc, then diverges "
                 "after the bounce. Middle: per-parameter marker sensitivity — density is 50× "
                 "below the launch velocities. Right: the post-bounce divergence in the x-z plane.")
    s += """
<p>The honest limit: even seven starts leave a ~18&nbsp;mm floor, because rigid contact
makes the loss landscape rugged — bounce timing is chaotic in the parameters, so local
optimization traps. This is the same ruggedness we saw far from the target in the cloth
case, now intrinsic to contact. The scientific takeaway is unchanged and is the whole
point of the project: <b>you can only recover what the motion observes, and the
identifiability analysis tells you in advance which parameters those are.</b></p>

<h2><span class="tag">M6</span> A real multi-asset scene — and a gauge that a collision can bend</h2>
<p>The last step leaves the empty stage behind: two real scanned objects — the teapot
and a great-white-shark scan — launched toward each other <b>on a real wooden table</b>,
colliding and coming to rest. We recover both objects' physics at once (twelve
parameters), and ask a question the single object could not: when two bodies collide,
does their momentum exchange finally reveal their masses?</p>
"""
    s += img_tag("scene_render.png",
                 "Two real scanned objects colliding on the real table mesh, in Newton's contact "
                 "solver. Object-object contact uses SDF collision; the tabletop is a box proxy "
                 "with the real mesh kept for rendering.")
    s += """
<div class="finding"><b>Best fit 13.9&nbsp;mm — sharper than the single object</b> (17.7&nbsp;mm),
and once again the recovered trajectories trace the true collision almost exactly. The
per-parameter accuracy follows the identifiability ranking as faithfully as before: the
launch velocities come back to ~15&nbsp;%, the contact materials are soft, and the two
<b>densities are the least-observable parameters of all</b> — the teapot's is 76&nbsp;%
off and the lighter shark's runs clean off to the clamp.</div>
<p><b>Does the collision break the density gauge?</b> It bends it. Swapping the two true
densities shifts the motion by 36&nbsp;mm, so the impact genuinely couples the masses —
yet locally the coupling is weak: the direction that trades density between the objects
is only 1.4× more observable than raising both together, and both sit ~20× below the
velocity signals. So the per-object gauge is <em>softened, not broken</em>. In a
launch-and-settle scene the kinematics dominate; to read relative density cleanly you
would need a collision-dominated one — repeated, harder impacts where the exchange of
momentum is the main event. The honest conclusion is the throughline of the whole
project, now proven across cloth, a single rigid body, and a multi-body scene: <b>you
recover exactly what the motion makes observable — no more, and the analysis says in
advance how much that is.</b></p>
"""
    s += img_tag("scene_recover.png",
                 "Left/right: recovered paths (dashed) track the true collision. Middle: the two "
                 "log-density bars are the smallest — density is the least-observable parameter, "
                 "even with an object-object collision to couple the masses.")
    s += """
<h2><span class="tag">M7</span> Off the white background — and a gauge you can break but not read</h2>
<p>Two threads close the current arc. First, the renders leave the void behind: our
cloth-sim flag now flies on a pole in a <b>real city street</b> — a painted wooden bench,
a Victorian lamp, a fire hydrant, and a photographed urban sky, lit and shadowed by
Blender's path tracer. The same flag geometry that was optimized against motion is now
staged in an environment, which is exactly the realistic initial frame the video-prior
pipeline wants.</p>
"""
    s += img_tag("city_scene.png",
                 "Our simulated flag flying in a real city scene: PolyHaven CC0 HDRI + street "
                 "furniture, rendered in Blender/Cycles. The flag's shape is the cloth-sim mesh.")
    s += """
<p>Second, we chased the density gauge into a corner. The multi-asset scene only
<em>bent</em> it; so we built a <b>collision-dominated</b> one — two comparable-mass
scanned objects meeting head-on, near-elastically, where the split of momentum is set by
their mass ratio. It worked, in the sense that matters for identifiability: density's
sensitivity leapt an order of magnitude, and the direction that trades mass between the
objects became <b>3.4× more observable</b> than scaling both together. The gauge, by every
local measure, is broken.</p>
<div class="finding"><b>And yet the recovery collapses.</b> Every Levenberg–Marquardt
start froze at its initial guess (best fit 224&nbsp;mm vs 13.9&nbsp;mm before); the
damping ran to 10⁸ on the first step because no descent direction exists. The very
violence that reveals density makes bounce timing chaotic in the parameters, so the
finite-difference gradient is noise. <b>Observability and optimizability are in direct
conflict:</b> the gentle scene is smooth but blind to density; the violent scene sees
density but is unsolvable by local methods. The escape is a global, gradient-free search
that tolerates a rough landscape — the result of that attempt is below.</p>
<div class="finding"><b>The global search fails too — and that is the finding.</b> CEM,
which needs no gradient and tolerates a rough surface, also never found the target: best
fit 1130&nbsp;mm, its loss frozen near its starting value for all sixteen iterations, and
the density ratio recovered as 0.52 against a true 2.0 — not just wrong in magnitude but
inverted. So the collision scene is unrecoverable by <em>either</em> local or global
optimization.</div>
<p>This closes the question cleanly. <b>Observability is not recoverability.</b> The
collision-dominated scene genuinely makes density observable — but the same violence that
reveals it turns the loss landscape into a needle in a fractal haystack: the true optimum
is a vanishingly small basin surrounded by deceptive shallow minima, and the dramatic
1.4&nbsp;m density-swap sensitivity is the <em>symptom</em> of that intractability, not a
promise of an easy fit. The bracket is now closed on both sides — the gentle scene is
smooth but blind to density, the violent scene sees density but is unsolvable — and for
this rigid-contact system there is no free lunch between them. Reading density off a
collision would take a different lever entirely: higher-time-resolution observation of the
impact, an analytic/smooth contact model, or fitting the momentum-conservation relation
directly instead of the full chaotic trajectory. It is the sharpest form of the lesson the
whole project keeps teaching — a parameter is recoverable only where it smoothly shapes the
motion, and violent contact is exactly where that smoothness breaks.</p>

<h3>High-resolution observation — and the root cause of every contact failure</h3>
<p>There was one more lever worth pulling: if the collision is a brief event we were
aliasing at 60&nbsp;fps, then observing it at ~1500&nbsp;fps should expose the velocity
change at contact — and the mass ratio is a clean, algebraic function of that change
(m₀Δv₀ = −m₁Δv₁), no chaotic trajectory fit required. So we threw two comparable-mass
scanned objects at each other <em>in mid-air</em>, where only their mutual impulse acts, and
resolved the impact.</p>
<div class="finding"><b>The observation works; the solver doesn't.</b> High resolution
resolves the impact cleanly — but the law the method rests on is broken by the simulator.
Measured against the true launch-and-separation velocities, the total horizontal momentum
<b>drops 57% through the collision</b>, when it must stay exactly constant. The leak is
identical at 18, 50, and 100 solver iterations, so it is fundamental to the position-based
contact model, not a convergence artifact — and it drags the recovered mass ratio to half
its true value.</div>
"""
    s += img_tag("momentum_nonconservation.png",
                 "Left: the mid-air collision cleanly exchanges horizontal velocity between the two "
                 "objects. Right: total horizontal momentum is flat in free flight, then collapses "
                 "57% during contact — XPBD's position-based contact does not conserve momentum.")
    s += """
<p>This is the thread that ties every contact result together. Newton's XPBD contact is
built for fast, stable <em>forward</em> simulation, not for the physical fidelity an inverse
problem needs — across the milestones it turned out to be <b>non-differentiable</b> (zero
gradient, M5), <b>non-scale-invariant</b> (absolute density leaks in, M7), and now
<b>non-momentum-conserving</b> (M7b). Any method that leans on the contact impulse being
correct — a gradient, or a momentum measurement — breaks on the same rock. The fix is not a
cleverer optimizer or a finer camera; it is a momentum-conserving contact solver (MuJoCo, or
an impulse/LCP formulation). The high-resolution observation is already right and sufficient;
with a faithful solver under it, reading density off a collision becomes a linear
measurement rather than an impossible fit.</p>
<h2>What's next</h2>
<ul>
<li><b>M4 stage 2</b>: swap the self-rendered video for real / generated footage —
requires choosing an image-to-video model (step 3) and likely upgrading LK to
CoTracker for real textures.</li>
<li>Scale: multi-object scenes and real asset data, once single-object M4 holds.</li>
<li>Formalize the eval harness (recovery error across seeds / scenes).</li>
</ul>

<h2>Reproduce</h2>
<pre>conda create -n warp python=3.11 -y
conda run -n warp pip install -r requirements.txt
bash scripts/fetch_assets.sh                       # real GSO / PolyHaven assets
python scripts/run_forward.py                      # M0
python scripts/check_grad.py                       # M1
python scripts/recover_full.py --method lm --fix log_mass --starts 5   # M3
python scripts/probe_identifiability.py            # M3 probe
python scripts/run_m4_pipeline.py --starts 3       # M4 stage 1
python scripts/run_rigid_forward.py                # M5 real-asset drop
python scripts/recover_rigid.py --start 0 --out o.json  # (one LM start; run several)
python scripts/recover_rigid.py --aggregate        # M5 merge + identifiability
python scripts/run_scene_forward.py --env table    # M6 multi-asset collision scene
python scripts/recover_scene.py --start 0 --out o.json  # (pack starts on ONE GPU)
python scripts/recover_scene.py --aggregate        # M6 merge + identifiability</pre>
<p><em>Technical log for the working agent: <code>docs/PROJECT_LOG.md</code>.</em></p>
"""
    if not artifact:
        s += "</body></html>\n"
    return s


M4_RESULT_DEFAULT = """
<div class="finding"><b>Result (3 starts, 333&nbsp;s).</b> All starts converge to the
same plateau: 2D loss ≈ 6×10⁻⁵ ≈ <b>4&nbsp;px RMS — the Lucas–Kanade tracking noise
floor</b>, not an optimizer failure. Against these noisy single-view tracks the
well-identified parameters recover to ~14–30&nbsp;% (wind a0 13.8&nbsp;%, gravity
19&nbsp;%, stretch 25&nbsp;%, damping 30&nbsp;%); sloppy directions degrade further.
Since the same optimizer reaches ≤0.07&nbsp;% with exact 3D supervision, the accuracy
limit has moved from <em>optimization</em> to <em>observation quality</em>.</div>
<table>
<tr><th>param</th><th>true</th><th>recovered</th><th>rel err</th></tr>
<tr><td>wind a0</td><td>15.0</td><td>12.94</td><td>13.8 %</td></tr>
<tr><td>wind a1</td><td>4.0</td><td>1.62</td><td>59.5 %</td></tr>
<tr><td>wind a2</td><td>−3.0</td><td>−0.05</td><td>98.2 %</td></tr>
<tr><td>gravity_z</td><td>−9.81</td><td>−7.91</td><td>19.3 %</td></tr>
<tr><td>tri_ke</td><td>5000</td><td>3745</td><td>25.1 %</td></tr>
<tr><td>tri_kd</td><td>10.0</td><td>6.96</td><td>30.4 %</td></tr>
<tr><td>edge_ke</td><td>10.0</td><td>0.0</td><td>unobservable</td></tr>
<tr><td>mass</td><td>0.05</td><td>0.05</td><td>fixed (gauge)</td></tr>
</table>
"""


if __name__ == "__main__":
    m4_html = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else M4_RESULT_DEFAULT
    path = REPO / "docs" / "report.html"
    path.write_text(build(m4_html))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")
    if len(sys.argv) > 2 and sys.argv[2] == "--artifact":
        apath = Path(sys.argv[3])
        apath.write_text(build(m4_html, artifact=True))
        print(f"wrote {apath} ({apath.stat().st_size / 1024:.0f} KiB)")
