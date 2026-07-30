"""Build docs/report.html from the measurement and audit output. Self-contained, no CDN.

Reads outputs/scene/fulllab/{simple_fit,audit_pixel}.json and renders every figure from
them, so the report cannot drift from the numbers the way the hand-maintained one did.

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
NICE = {"apple": "apple", "baseball": "baseball", "brass_pot": "brass pot",
        "ceramic_vase": "ceramic vase", "rubber_duck": "rubber duck"}
OLD_GRID = {"baseball": 0.680, "brass_pot": 0.365, "ceramic_vase": 0.470, "apple": 1.100}
STATES = ["MOVES", "STATIC", "DEGRADED"]
SCOL = {"MOVES": GREEN, "STATIC": LINE, "DEGRADED": RED}


def style(ax):
    ax.set_facecolor(PLATE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=LINE, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def embed(fig, photo=False):
    """PNG for charts (flat colour, compresses well); JPEG for photographic frames.

    The degrading-brass-pot filmstrip is 14 rendered photographs and was 798 KB as PNG,
    four times the rest of the report combined.
    """
    buf = io.BytesIO()
    if photo:
        fig.savefig(buf, format="jpeg", dpi=110, facecolor=PLATE, bbox_inches="tight",
                    pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    fig.savefig(buf, format="png", dpi=150, facecolor=PLATE, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_two_estimators(d):
    rows = [(o, d[o]["slide"]) for o in d
            if d[o].get("slide", {}).get("value") is not None]
    rows.sort(key=lambda r: r[1]["n"])
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
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig)


def fig_audit(a):
    """MOVES / STATIC / DEGRADED per probe, from the appearance-based instrument."""
    pk = a["per_kind"]
    kinds = ["drop", "slide", "collide"]
    fig, ax = plt.subplots(figsize=(8.4, 2.35))
    style(ax)
    y = np.arange(len(kinds))
    left = np.zeros(len(kinds))
    for s in STATES:
        w = np.array([pk.get(k, {}).get(s, 0) for k in kinds], float)
        ax.barh(y, w, left=left, color=SCOL[s], height=0.6,
                label={"MOVES": "moves, asset intact", "STATIC": "genuinely static",
                       "DEGRADED": "asset stops being itself"}[s])
        left += w
    ax.set_yticks(y); ax.set_yticklabels(kinds, fontsize=9)
    ax.set_ylim(len(kinds) - 0.45, -0.55)
    ax.set_xlabel("takes", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig)


def fig_intervals(d):
    objs = list(d.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))
    for ax, (kind, name, sym, rng) in zip(axes, PARAMS):
        style(ax)
        for i, o in enumerate(objs):
            r = d[o].get(kind) or {}
            if r.get("value") is None:
                ax.hlines(i, rng[0], rng[1], color=LINE, lw=1.0, ls=(0, (3, 3)), zorder=1)
                continue
            v, e = r["value"], r["interval"]
            col = AMBER if r.get("single_take") else INK
            ax.errorbar([v], [i], xerr=[[min(e, v - rng[0] + 1e-9)], [e]], fmt="o",
                        ms=6, color=col, ecolor=col, elinewidth=1.7, capsize=3.5, zorder=3)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([NICE.get(o, o) for o in objs] if ax is axes[0] else [],
                           fontsize=8.5)
        ax.set_ylim(len(objs) - 0.45, -0.55)
        ax.set_xlim(rng[0] - 0.03 * (rng[1] - rng[0]), rng[1])
        ax.set_title(f"{name}   {sym}", color=INK, fontsize=10, loc="left", pad=6)
    axes[0].plot([], [], color=LINE, ls=(0, (3, 3)), label="no measurement")
    axes[0].plot([], [], "o", color=AMBER, ms=5, label="single take (uncertainty unknown)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
               labelcolor=MUTED, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.07))
    return embed(fig)


def fig_degraded():
    """The brass pot ceasing to be a brass pot -- the finding the old audit could not see."""
    picks = [("brass_pot_slide_seed0", "slide"), ("brass_pot_drop_seed2", "drop")]
    idx = [0, 8, 16, 24, 32, 40, 48]
    fig, axes = plt.subplots(len(picks), len(idx),
                             figsize=(1.62 * len(idx), 1.55 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (clip, lab) in enumerate(picks):
        p = LAB / f"vid_{clip}.npz"
        if not p.exists():
            continue
        fr = np.load(p)["frames"]
        for c, fi in enumerate(idx):
            ax = axes[r, c]; ax.imshow(fr[min(fi, len(fr) - 1)])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"f{fi}", fontsize=8, pad=2)
        axes[r, 0].set_ylabel(lab, fontsize=9, rotation=0, ha="right", va="center",
                              color=INK)
    fig.tight_layout()
    return embed(fig, photo=True)


def main():
    d = json.loads((LAB / "simple_fit.json").read_text())
    a = json.loads((LAB / "audit_pixel.json").read_text())
    c = a["counts"]
    total = sum(c.values())
    moves, static, degraded = c.get("MOVES", 0), c.get("STATIC", 0), c.get("DEGRADED", 0)

    f_two, f_aud, f_int, f_deg = (fig_two_estimators(d), fig_audit(a),
                                  fig_intervals(d), fig_degraded())
    nparam = sum(len(o) for o in d.values())
    usable = sum(1 for o in d.values() for r in o.values()
                 if r.get("value") is not None
                 and str(r.get("verdict", "")).startswith("usable"))

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
figcaption {{ font-size:.845rem; color:var(--muted); margin-top:.6rem; max-width:44rem; }}
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
td .pm {{ color:var(--muted); }} td.none {{ color:var(--muted); }}
.pill {{ font:600 9.5px/1.5 ui-monospace,monospace; letter-spacing:.06em;
  text-transform:uppercase; color:var(--warn); border:1px solid var(--warn);
  border-radius:3px; padding:0 .3rem; vertical-align:1px; }}
.call {{ border-left:3px solid var(--accent); background:var(--card);
  padding:1rem 1.15rem; border-radius:0 9px 9px 0; margin:1.4rem 0; }}
.call.bad {{ border-left-color:var(--bad); }}
.call.warn {{ border-left-color:var(--warn); }}
.call b {{ color:var(--ink); }} .call p {{ margin:.35rem 0 0; }}
code {{ font:.87em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--card); padding:.1em .35em; border-radius:4px; }}
ul {{ max-width:44rem; padding-left:1.15rem; }} li {{ margin:.35rem 0; }}
del {{ color:var(--muted); text-decoration-color:var(--bad); }}
footer {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--line);
  font-size:.82rem; color:var(--muted); }}
</style>

<div class="wrap">
<div class="eyebrow">Per-instance inverse simulation · measurement report</div>
<h1>Three times, the instrument turned out to be the thing being measured</h1>
<p class="lede">We stage physics experiments, film them with a video model, and recover
material parameters from the motion. Every layer of that pipeline — the estimator, its
acceptance gate, and the point tracker underneath both — has now been caught reporting
its own failure as a property of the video. This report is what survived.</p>

<div class="headline">
  <div><b>{moves} / {total}</b><span>takes do contain motion, measured without a
    tracker — correcting an earlier claim that most contained none</span></div>
  <div><b>{degraded} / {total}</b><span>takes where the object stops being the object,
    a failure no rigid-body parameter can explain</span></div>
  <div><b>{usable} / {nparam}</b><span>parameters pinned down — but see the caveat on
    what these still rest on</span></div>
</div>

<h2><span class="num">01</span>The estimator was manufacturing numbers</h2>
<p>The retired fitter searched a parameter grid for the value whose hand-weighted motion
signature best matched the track. A grid always returns a best match; it never asks
whether the observation constrains anything. Measuring the physics directly asks that
first. This comparison runs both estimators over <em>identical</em> tracks, so it is
unaffected by everything that follows.</p>
<figure><img src="{f_two}" alt="Per-take friction versus the grid fitter's answer">
<figcaption>Each dot is one take's friction, measured from its deceleration. The
baseball's ten takes span 0.007–0.285. The retired grid fitter reported <b>0.680</b>
from those same ten takes — outside the range any individual take supports. The apple's
1.100 was the grid maximum, a railed bound reported as a measurement.</figcaption>
</figure>

<h2><span class="num">02</span>The rebuilt estimator: measure, don't search</h2>
<p>Each probe is analytically invertible, so nothing is searched:</p>
<ul>
<li><b>drop</b> → restitution <code>e = |v_up| / |v_down|</code> — dimensionless</li>
<li><b>slide</b> → friction <code>μ = |a| / g</code> — the only one needing the px↔m scale</li>
<li><b>collide</b> → <code>m_target/m_mover = (v_pre − v_post) / v_target</code> —
  scale and frame rate cancel exactly</li>
</ul>
<p>Every quantity is a velocity from a least-squares fit over about five frames, which
yields a standard error for free. Uncertainty is propagated rather than guessed, so each
parameter arrives as a value with an interval — and an interval spanning the plausible
range <em>is</em> the &ldquo;not identifiable&rdquo; verdict. Every tunable is gone: no
grids, no feature weights, no scoring variants, no agreement threshold.</p>
<div class="tw"><table>
<thead><tr><th>object</th><th>restitution e</th><th>friction μ</th>
<th>mass ratio m<sub>t</sub>/m<sub>m</sub></th></tr></thead>
<tbody>{rows}</tbody></table></div>
<figure><img src="{f_int}" alt="Interval plot for every parameter">
<figcaption>Every parameter with its propagated interval. No handbook comparison is
drawn: a downloaded mesh has no true density, and the goal is a parameter that
<em>explains the clip</em>, not one that matches a reference book.</figcaption></figure>
<div class="call warn"><b>Caveat that outranks the table.</b>
<p>These values are computed from CoTracker point tracks, and section 03 shows that
tracking layer is unreliable on these clips. The 0-of-15 result should be read as
&ldquo;not established&rdquo;, not as &ldquo;proven unmeasurable&rdquo;. Re-deriving
them on validated tracks is the outstanding work.</p></div>

<h2><span class="num">03</span>The audit that corrected itself</h2>
<p>An earlier version of this report stated that <del>68 of 93 takes contained no
measurable motion</del>. That was wrong. It was computed from tracked centroids, and the
tracker fails on these clips in a way its own diagnostics cannot see: points stay
&ldquo;visible&rdquo; while sitting on the background, so a duck that visibly fell to the
table measured 6&nbsp;px of motion. 38% of the takes behind that claim had a suspect
tracker.</p>
<p>Re-run with an independent, appearance-based instrument — normalised cross-correlation
against the object's own patch from frame 0, which reports <em>where</em> it matches and
<em>how well</em> — the picture inverts:</p>
<figure><img src="{f_aud}" alt="Audit of every take by probe">
<figcaption>{moves} of {total} takes move with the asset intact; only {static} are
genuinely static. The motion was there all along.</figcaption></figure>

<h2><span class="num">04</span>The finding the old audit could not see</h2>
<p>The appearance channel adds a state that centroid tracking had no way to express: the
object stops being the object. All {degraded} such takes are the brass pot, which
transforms mid-clip from a lidded pot into a wide shallow bowl.</p>
<figure><img src="{f_deg}" alt="Brass pot degrading across two clips">
<figcaption>The brass pot during a slide (top) and a drop (bottom). This is not a
tracking artefact — I suspected it was one, given the pot's low-texture specular surface,
and the frames say otherwise.</figcaption></figure>
<div class="call bad"><b>No θ explains this.</b>
<p>A parameter that merely has to produce a <em>plausible</em> physical motion is a
generous bar, and asset degradation fails it outright: there is no density, friction or
restitution that turns a pot into a bowl. This is not identifiability and not frame rate.
It is asset integrity, and it is a different problem from the one we were solving.</p>
</div>

<h2><span class="num">05</span>Corrections</h2>
<ul>
<li><b>&ldquo;The videos do not contain the experiment&rdquo; was an instrument
artefact.</b> Stated twice, on tracker evidence. {moves} of {total} takes move.</li>
<li><b>60&nbsp;fps was not the diagnosis for mass</b>, and is no longer proposed as a
requirement. The per-take fits were individually decisive and mutually contradictory,
which is generator non-repeatability, not temporal resolution.</li>
<li><b>There is no ground truth in this pipeline.</b> Errors were once quoted against
&ldquo;true&rdquo; densities from a hardcoded table of textbook guesses, anchored by a
value from that same table.</li>
<li><b>A max-excursion statistic is not a displacement.</b> The first appearance-based
audit used peak displacement and scored 38&nbsp;px of &ldquo;motion&rdquo; for a vase
that never left its spot, because it transiently splits into two blobs and re-forms.
Switching to end-to-end displacement moved six takes from moving to static — the same
mistake as the original path-length gate, made a second time.</li>
</ul>

<h2><span class="num">06</span>Known limits of the current instrument</h2>
<ul>
<li>One clip (<code>rubber_duck_collide_seed0</code>) is classed as moving at 0.55
object-widths where visual inspection says the duck re-forms rather than translates. The
appearance tracker cannot fully separate &ldquo;translated&rdquo; from
&ldquo;re-rendered nearby&rdquo;.</li>
<li>The audit is calibrated against roughly a dozen clips inspected frame by frame, not
against all {total}.</li>
<li>Parameter values still depend on point tracks that have not been re-validated.</li>
</ul>

<footer>Generated by <code>scripts/make_report.py</code> from
<code>simple_fit.json</code> and <code>audit_pixel.json</code>. Full chronology,
including every retraction, in <code>docs/PROJECT_LOG.md</code>.</footer>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB)")
    print(f"  audit: {moves} moves / {static} static / {degraded} degraded of {total}")
    print(f"  params: {usable}/{nparam} usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
