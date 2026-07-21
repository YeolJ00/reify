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


HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inverse Simulation from Video Priors — M0–M4 Report</title>
<style>
  :root { --fg:#1a1a1a; --bg:#ffffff; --accent:#0b5fa5; --soft:#f4f6f8; --line:#d8dee4; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --bg:#14181c; --accent:#6ab0e8; --soft:#1d242b; --line:#333c45; }
  }
  body { font: 16px/1.6 system-ui, sans-serif; color:var(--fg); background:var(--bg);
         max-width: 900px; margin: 0 auto; padding: 2rem 1.2rem 4rem; }
  h1 { font-size: 1.7rem; border-bottom: 2px solid var(--accent); padding-bottom:.4rem; }
  h2 { font-size: 1.25rem; color: var(--accent); margin-top: 2.2rem; }
  h3 { font-size: 1.05rem; margin-top: 1.6rem; }
  code, pre { font-family: ui-monospace, monospace; background: var(--soft);
              border-radius: 4px; }
  code { padding: .1em .35em; font-size: .88em; }
  pre { padding: .8rem 1rem; overflow-x: auto; font-size: .82em; line-height: 1.45; }
  table { border-collapse: collapse; margin: 1rem 0; font-size: .9em; width: 100%; }
  th, td { border: 1px solid var(--line); padding: .35rem .6rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: var(--soft); }
  figure { margin: 1.2rem 0; text-align: center; }
  figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }
  figcaption { font-size: .85em; opacity: .75; margin-top: .4rem; }
  .finding { background: var(--soft); border-left: 4px solid var(--accent);
             padding: .7rem 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0; }
  .missing { color: #c0392b; }
  .tag { display:inline-block; background:var(--accent); color:var(--bg);
         border-radius: 10px; padding: 0 .55em; font-size:.75em; font-weight:600;
         vertical-align: middle; margin-right:.4em; }
</style></head><body>
"""


def build(m4_result_html: str) -> str:
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
python scripts/run_forward.py                     # M0
python scripts/check_grad.py                      # M1
python scripts/recover_full.py --method lm --fix log_mass --starts 5   # M3
python scripts/probe_identifiability.py           # M3 probe
python scripts/run_m4_pipeline.py --starts 3      # M4 stage 1</pre>
<p><em>Technical log for the working agent: <code>docs/PROJECT_LOG.md</code>.</em></p>
</body></html>
"""
    return s


if __name__ == "__main__":
    m4_html = sys.argv[1] if len(sys.argv) > 1 else "<p class='missing'>[M4 results pending]</p>"
    path = REPO / "docs" / "report.html"
    path.write_text(build(m4_html))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")
