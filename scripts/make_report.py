"""Build docs/report.html from the measurement output. Self-contained, no CDN.

Reads outputs/scene/fulllab/simple_fit.json (written by scripts/simple_fit.py) and
renders the figures from it, so the report cannot drift from the numbers the way the
previous hand-maintained one did — it described a pipeline that no longer existed.

Run: python scripts/make_report.py            (warp env, no GPU)
"""
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LAB = REPO / "outputs" / "scene" / "fulllab"
OUT = REPO / "docs" / "report.html"

INK, MUTED, LINE = "#141a22", "#5d6b7a", "#cbd5e0"
CYAN, AMBER, RED, GREEN = "#0e7c93", "#b06d00", "#b3261e", "#2f6b3a"
PLATE = "#f7f9fb"

PARAMS = [("drop", "restitution", "e", (0.0, 1.0)),
          ("slide", "friction", "μ", (0.0, 1.2)),
          ("collide", "mass ratio", "m$_t$/m$_m$", (0.0, 2.0))]
# Typical handbook values for these materials. Shown as a REFERENCE BAND only: these
# are expectations, not ground truth. A downloaded mesh has no true density, and
# quoting error against such numbers is a mistake this project already made once.
EXPECTED = {"drop": (0.10, 0.50), "slide": (0.30, 0.60), "collide": None}
NICE = {"apple": "apple", "baseball": "baseball", "brass_pot": "brass pot",
        "ceramic_vase": "ceramic vase", "rubber_duck": "rubber duck"}

# What the retired grid fitter reported for the same takes, for the comparison figure.
# Kept explicit rather than recomputed: those scripts are deleted (see PROJECT_LOG M39).
OLD_GRID = {"baseball": 0.680, "brass_pot": 0.365, "ceramic_vase": 0.470, "apple": 1.100}


def style(ax):
    ax.set_facecolor(PLATE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, axis="both", color=LINE, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def embed(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=PLATE, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def embed_file(p, max_w=1500):
    if not Path(p).exists():
        return None
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_two_estimators(d):
    """The single most important figure: per-take spread vs what the grid reported."""
    rows = [(o, d[o]["slide"]) for o in d
            if d[o].get("slide", {}).get("value") is not None]
    rows.sort(key=lambda r: r[1]["n"])                 # largest n at the top
    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(rows) + 1.35))
    style(ax)
    for i, (o, r) in enumerate(rows):
        sm = np.array(r["samples"], float)
        ax.hlines(i, sm.min(), sm.max(), color=CYAN, lw=1.4, alpha=0.4, zorder=2)
        ax.plot(sm, [i] * len(sm), "o", ms=6.5, mfc=CYAN, mec="white", mew=1.0,
                zorder=3, label="individual take" if i == 0 else None)
        ax.plot([r["value"]], [i], "D", ms=8, color=INK, zorder=4,
                label="combined measurement" if i == 0 else None)
        if o in OLD_GRID:
            ax.plot([OLD_GRID[o]], [i], "X", ms=11, color=RED, zorder=5,
                    label="retired grid fitter" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NICE.get(o,o)}  (n={r['n']})" for o, r in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("friction μ measured from deceleration", color=MUTED, fontsize=9)
    ax.set_xlim(-0.03, 1.20)
    # legend ABOVE the axes: inside it overlapped the densest row
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig)


def fig_intervals(d):
    """Forest plot: every parameter with its interval, against handbook expectation."""
    objs = list(d.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.15))
    for ax, (kind, name, sym, rng) in zip(axes, PARAMS):
        style(ax)
        exp = EXPECTED.get(kind)
        if exp:
            ax.axvspan(exp[0], exp[1], color=GREEN, alpha=0.13, zorder=0, lw=0)
        for i, o in enumerate(objs):
            r = d[o].get(kind) or {}
            if r.get("value") is None:
                # a dashed rule instead of text: the italic label used to collide
                # with the axis and with neighbouring panels
                ax.hlines(i, rng[0], rng[1], color=LINE, lw=1.0, ls=(0, (3, 3)),
                          zorder=1)
                continue
            v, e = r["value"], r["interval"]
            single = bool(r.get("single_take"))
            col = AMBER if single else INK
            ax.errorbar([v], [i], xerr=[[min(e, v - rng[0] + 1e-9)], [e]], fmt="o",
                        ms=6, color=col, ecolor=col, elinewidth=1.7, capsize=3.5,
                        zorder=3)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([NICE.get(o, o) for o in objs] if ax is axes[0] else [],
                           fontsize=8.5)
        ax.set_ylim(len(objs) - 0.45, -0.55)           # padding, and top-down order
        ax.set_xlim(rng[0] - 0.03 * (rng[1] - rng[0]), rng[1])
        ax.set_title(f"{name}   {sym}", color=INK, fontsize=10, loc="left", pad=6)
    axes[0].plot([], [], color=LINE, ls=(0, (3, 3)), label="no measurement")
    axes[0].plot([], [], "o", color=AMBER, ms=5, label="single take")
    axes[0].add_patch(plt.Rectangle((0, 0), 0, 0, color=GREEN, alpha=0.13,
                                    label="handbook range (expectation, not truth)"))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
               labelcolor=MUTED, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    return embed(fig)


def fig_audit(d):
    """Where the 93 takes went."""
    objs = list(d.keys())
    meas = [sum((d[o].get(k) or {}).get("n", 0) or 0 for k, *_ in PARAMS) for o in objs]
    nomo = [sum((d[o].get(k) or {}).get("n_rejected", 0) or 0 for k, *_ in PARAMS)
            for o in objs]
    cont = [sum((d[o].get(k) or {}).get("n_contradictory", 0) or 0 for k, *_ in PARAMS)
            for o in objs]
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(objs) + 1.3))
    style(ax)
    y = np.arange(len(objs))
    ax.barh(y, meas, color=GREEN, label="yielded a measurement", height=0.62)
    ax.barh(y, nomo, left=meas, color=LINE, label="no measurable motion", height=0.62)
    ax.barh(y, cont, left=np.array(meas) + np.array(nomo), color=RED,
            label="contradicted physics", height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(o, o) for o in objs], fontsize=9)
    ax.set_xlabel("takes", color=MUTED, fontsize=9)
    ax.set_ylim(len(objs) - 0.45, -0.55)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig), sum(meas), sum(nomo), sum(cont)


def main():
    d = json.loads((LAB / "simple_fit.json").read_text())
    f_two = fig_two_estimators(d)
    f_int = fig_intervals(d)
    f_aud, n_meas, n_nomo, n_cont = fig_audit(d)
    total = n_meas + n_nomo + n_cont
    strip = embed_file(LAB / "inspect_ceramic_vase_collide.png")
    hero = embed_file(REPO / "outputs" / "scene" / "hero.png", 1200)

    usable = sum(1 for o in d.values() for r in o.values()
                 if r.get("value") is not None
                 and str(r.get("verdict", "")).startswith("usable"))
    nparam = sum(len(o) for o in d.values())

    rows = ""
    for o in d:
        cells = ""
        for kind, _n, _s, _r in PARAMS:
            r = d[o].get(kind) or {}
            if r.get("value") is None:
                cells += '<td class="none">—</td>'
            else:
                tag = ' <span class="pill">1 take</span>' if r.get("single_take") else ""
                cells += (f'<td>{r["value"]:.3f} <span class="pm">± '
                          f'{r["interval"]:.3f}</span>{tag}</td>')
        rows += f"<tr><th>{NICE.get(o,o)}</th>{cells}</tr>"

    html = f"""<title>Inverse Simulation from Video — Measurement Report</title>
<style>
:root {{
  --ink:#141a22; --body:#33414f; --muted:#6b7a89; --line:#dde4ea;
  --bg:#ffffff; --card:#f7f9fb; --accent:#0e7c93; --warn:#b06d00; --bad:#b3261e;
  --plate:#f7f9fb;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e8eef4; --body:#b9c6d2; --muted:#8b9aa9; --line:#26313d;
    --bg:#0e141a; --card:#161f28; --accent:#4fc3d9; --warn:#e0a340; --bad:#f2836f; }}
}}
:root[data-theme="dark"] {{ --ink:#e8eef4; --body:#b9c6d2; --muted:#8b9aa9;
  --line:#26313d; --bg:#0e141a; --card:#161f28; --accent:#4fc3d9;
  --warn:#e0a340; --bad:#f2836f; }}
:root[data-theme="light"] {{ --ink:#141a22; --body:#33414f; --muted:#6b7a89;
  --line:#dde4ea; --bg:#ffffff; --card:#f7f9fb; --accent:#0e7c93;
  --warn:#b06d00; --bad:#b3261e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--body);
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:62rem; margin:0 auto; padding:3.2rem 1.4rem 5rem; }}
.eyebrow {{ font:600 11.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent); }}
h1 {{ font-size:clamp(1.85rem,4vw,2.7rem); line-height:1.12; margin:.7rem 0 .5rem;
  color:var(--ink); letter-spacing:-.02em; text-wrap:balance; font-weight:650; }}
.lede {{ font-size:1.12rem; color:var(--body); max-width:44rem; margin:0 0 2.2rem; }}
h2 {{ font-size:1.28rem; color:var(--ink); margin:3rem 0 .5rem; letter-spacing:-.01em;
  text-wrap:balance; font-weight:640; }}
h2 .num {{ font:600 12px/1 ui-monospace,monospace; color:var(--accent);
  letter-spacing:.1em; display:block; margin-bottom:.35rem; }}
p {{ max-width:44rem; }}
figure {{ margin:1.5rem 0; }}
figure img {{ width:100%; display:block; border:1px solid var(--line);
  border-radius:9px; background:var(--plate); }}
figcaption {{ font-size:.845rem; color:var(--muted); margin-top:.6rem;
  max-width:44rem; }}
.headline {{ display:flex; flex-wrap:wrap; gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:11px; overflow:hidden; margin:2rem 0; }}
.headline div {{ flex:1 1 12rem; background:var(--card); padding:1.15rem 1.25rem; }}
.headline b {{ display:block; font:700 1.85rem/1.05 ui-sans-serif,system-ui;
  color:var(--ink); font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
.headline span {{ font-size:.815rem; color:var(--muted); }}
.tw {{ overflow-x:auto; margin:1.4rem 0; }}
table {{ border-collapse:collapse; width:100%; min-width:34rem; font-size:.9rem; }}
th,td {{ text-align:left; padding:.6rem .8rem; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; }}
thead th {{ font:600 11.5px/1.3 ui-monospace,monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); }}
tbody th {{ font-weight:560; color:var(--ink); }}
td .pm {{ color:var(--muted); }}
td.none {{ color:var(--muted); }}
.pill {{ font:600 9.5px/1.5 ui-monospace,monospace; letter-spacing:.06em;
  text-transform:uppercase; color:var(--warn); border:1px solid var(--warn);
  border-radius:3px; padding:0 .3rem; vertical-align:1px; }}
.call {{ border-left:3px solid var(--accent); background:var(--card);
  padding:1rem 1.15rem; border-radius:0 9px 9px 0; margin:1.4rem 0; }}
.call.bad {{ border-left-color:var(--bad); }}
.call b {{ color:var(--ink); }}
.call p {{ margin:.35rem 0 0; }}
code {{ font:.87em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--card); padding:.1em .35em; border-radius:4px; }}
ul {{ max-width:44rem; padding-left:1.15rem; }}
li {{ margin:.35rem 0; }}
footer {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--line);
  font-size:.82rem; color:var(--muted); }}
</style>

<div class="wrap">
<div class="eyebrow">Per-instance inverse simulation · measurement report</div>
<h1>Nothing in this scene is currently measurable from generated video</h1>
<p class="lede">We stage physics experiments, film them with a video model, and recover
material parameters by measuring the motion. This report covers the point at which the
instrument was turned on itself: the estimator was reporting confident numbers for clips
in which nothing moved.</p>

<div class="headline">
  <div><b>{usable} / {nparam}</b><span>parameters measured with an interval tighter
    than ±25% from more than one take</span></div>
  <div><b>{n_nomo} / {total}</b><span>takes contained no measurable motion</span></div>
  <div><b>{n_cont}</b><span>takes contradicted physics outright — a mover that gains
    speed through impact</span></div>
</div>

<h2><span class="num">01</span>The estimator was manufacturing numbers</h2>
<p>The retired fitter searched a parameter grid for the value whose hand-weighted motion
signature best matched the track. A grid always returns a best match; it never asks
whether the observation constrains anything. Measuring the physics directly asks that
first.</p>
<figure><img src="{f_two}" alt="Per-take friction values versus the grid fitter's answer">
<figcaption>Each dot is one take's friction, measured from its deceleration. The
baseball's ten takes span 0.007–0.285. The retired grid fitter reported <b>0.680</b> from
those same ten takes — outside the range any individual take supports. The apple's
1.100 was the grid maximum, a railed bound reported as a measurement.</figcaption></figure>

<h2><span class="num">02</span>Why it went unnoticed: path length integrates noise</h2>
<p>The old "did it move" guard measured <em>path length</em> — the sum of per-frame
steps. That accumulates tracking jitter without bound: 49 frames of ±2&nbsp;px noise
manufactures roughly 100&nbsp;px of apparent travel from a stationary object. Net
displacement does not accumulate, because noise cancels.</p>
<div class="call bad"><b>Over 55 collision clips, the ceramic vase's subject never
displaced more than 21&nbsp;px and the rubber duck never more than 17&nbsp;px</b>
<p>— against the 23–43&nbsp;px they had to cover to reach their partner. The videos did
not contain the experiment, and the fits were reading sub-10-pixel jitter.</p></div>
{'<figure><img src="' + strip + '" alt="Filmstrip of two vase collision clips"><figcaption>Two ceramic-vase &ldquo;collisions&rdquo;, every seventh frame, with the tracked subject (cyan) and partner (magenta). The vase does not move. Both clips previously produced confident fits, 15.8&times; apart.</figcaption></figure>' if strip else ''}

<h2><span class="num">03</span>The rebuilt estimator: measure, don't search</h2>
<p>Each probe is analytically invertible, so nothing is searched:</p>
<ul>
<li><b>drop</b> → restitution <code>e = |v_up| / |v_down|</code> — dimensionless</li>
<li><b>slide</b> → friction <code>μ = |a| / g</code> — the only one needing the px↔m
  scale</li>
<li><b>collide</b> → <code>m_target/m_mover = (v_pre − v_post) / v_target</code> —
  scale and frame rate cancel exactly</li>
</ul>
<p>Every quantity is a velocity from a least-squares fit over about five frames, which
yields a standard error for free. Uncertainty is propagated rather than guessed, so each
parameter arrives as a value with an interval — and an interval spanning the plausible
range <em>is</em> the &ldquo;not identifiable&rdquo; verdict. Every tunable is gone: no
grids, no feature weights, no scoring variants, no agreement threshold. The
&ldquo;did it move&rdquo; test became a significance test on the fitted velocity.</p>

<h2><span class="num">04</span>What the scene actually yields</h2>
<div class="tw"><table>
<thead><tr><th>object</th><th>restitution e</th><th>friction μ</th>
<th>mass ratio m<sub>t</sub>/m<sub>m</sub></th></tr></thead>
<tbody>{rows}</tbody></table></div>
<figure><img src="{f_int}" alt="Interval plot for every parameter">
<figcaption>Every parameter with its propagated interval. The green band is the
handbook range for these materials — an <em>expectation</em>, not ground truth: a
downloaded mesh has no true density. Amber marks a single take, where there is no
repeatability term and the interval is a lower bound on the real uncertainty. Every
measured friction sits an order of magnitude below the handbook band, which is the
rolling-resistance problem noted below.</figcaption></figure>
<figure><img src="{f_aud}" alt="Take audit by object">
<figcaption>Of {total} takes, {n_meas} yielded a measurement. The rubber duck is inert in
every experiment — it never moves in any clip.</figcaption></figure>

<div class="call"><b>A probe-design error this exposed.</b>
<p>The μ ≈ 0.01–0.09 readings are about right for the <b>rolling resistance</b> of a ball
on wood — which is not the Coulomb μ the simulator consumes. A sphere's deceleration
cannot measure sliding friction. The slide probe is valid only for objects that actually
slide, so the baseball and apple need a different experiment or a rolling model.</p></div>

<h2><span class="num">05</span>Corrections to earlier reporting</h2>
<ul>
<li><b>60&nbsp;fps was not the diagnosis for mass.</b> The per-take fits were
individually decisive and mutually contradictory, which is generator non-repeatability,
not temporal resolution. Frame rate cannot be the binding constraint while half the clips
contain no motion.</li>
<li><b>There is no ground truth in this pipeline.</b> Errors were previously quoted
against &ldquo;true&rdquo; densities that came from a hardcoded table of textbook
guesses, anchored by a reference value taken from that same table. Downloaded meshes have
no true density.</li>
<li><b>Three &ldquo;established&rdquo; friction values did not survive.</b> Direct
measurement does not reproduce any of them.</li>
</ul>

<h2><span class="num">06</span>What this says about the next step</h2>
<p>The binding constraint is data collection, not estimation. Before designing further
probes, the generator needs screening: produce a few clips per candidate experiment and
measure whether the intended object moved at all. Static-equilibrium probes — flotation
for absolute density against water, a balance beam for mass ratio — would remove both the
frame-rate sensitivity and the arbitrary density anchor, but they still assume the model
animates the object, which for the vase and duck it does not.</p>

<footer>Generated by <code>scripts/make_report.py</code> from
<code>outputs/scene/fulllab/simple_fit.json</code>. Full chronology, including every
retraction, in <code>docs/PROJECT_LOG.md</code> (M39).</footer>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB)")
    print(f"  {usable}/{nparam} parameters usable, {n_meas}/{total} takes yielded a value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
