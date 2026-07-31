"""Build docs/report.html from the measurement, audit and comparison output.

Self-contained: every figure and animation is embedded, no external requests.
Reads outputs/scene/{fulllab,expand}/*.json and docs/cmp_*.webp.

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
EXP = REPO / "outputs" / "scene" / "expand"
DOCS = REPO / "docs"
OUT = DOCS / "report.html"

INK, MUTED, LINE = "#141a22", "#5d6b7a", "#cbd5e0"
CYAN, AMBER, RED, GREEN = "#0e7c93", "#b06d00", "#b3261e", "#2f6b3a"
PLATE = "#f7f9fb"
NICE = {"apple": "apple", "baseball": "baseball", "brass_pot": "brass pot",
        "ceramic_vase": "ceramic vase", "rubber_duck": "rubber duck",
        "wooden_bowl": "wooden bowl", "book": "book"}
OLD_GRID = {"baseball": 0.680, "brass_pot": 0.365, "ceramic_vase": 0.470, "apple": 1.100}
STATES = ["MOVES", "STATIC", "DEGRADED"]
SCOL = {"MOVES": GREEN, "STATIC": LINE, "DEGRADED": RED}
PRETTY_CMP = {"ceramic_vase_drop_mid": "ceramic vase · drop",
              "ceramic_vase_slide": "ceramic vase · slide",
              "rubber_duck_drop_mid": "rubber duck · drop"}
AXIS_NOTE = True


def style(ax):
    ax.set_facecolor(PLATE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=LINE, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def embed(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=PLATE, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def embed_path(p, mime):
    p = Path(p)
    if not p.exists():
        return None
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def fig_precision(fit):
    rows = []
    for subj in sorted(fit):
        r = fit[subj]
        if "e_mid" in r and r["n_drop"] > 1:
            rows.append((f"{NICE.get(subj,subj)} · restitution", r["e_mid"],
                         r["e_mid_se"], r["n_drop"]))
        c = r.get("friction")
        if c and c["n"] > 1:
            rows.append((f"{NICE.get(subj,subj)} · friction", c["value"],
                         c["interval"], c["n"]))
    rows.sort(key=lambda x: x[2] / max(abs(x[1]), 1e-9))
    fig, ax = plt.subplots(figsize=(8.6, 0.44 * len(rows) + 1.4))
    style(ax)
    for i, (lab, v, e, n) in enumerate(rows):
        rel = e / max(abs(v), 1e-9)
        col = GREEN if rel < 0.25 else (AMBER if rel < 0.5 else MUTED)
        ax.errorbar([v], [i], xerr=[[min(e, v)], [e]], fmt="o", ms=6, color=col,
                    ecolor=col, elinewidth=1.7, capsize=3.5, zorder=3)
        ax.text(0.995, i, f"±{100*rel:.0f}%  n={n}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.6, color=col)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-0.01, max(v + e for _l, v, e, _n in rows) * 1.45)
    ax.set_xlabel("value (restitution dimensionless; friction is μ)",
                  color=MUTED, fontsize=8.5)
    ax.plot([], [], "o", color=GREEN, ms=6, label="within ±25%")
    ax.plot([], [], "o", color=AMBER, ms=6, label="±25–50%")
    ax.plot([], [], "o", color=MUTED, ms=6, label="worse than ±50%")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig)


def fig_two_estimators(d):
    rows = [(o, d[o]["slide"]) for o in d
            if d[o].get("slide", {}).get("value") is not None]
    rows.sort(key=lambda r: r[1]["n"])
    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(rows) + 1.35))
    style(ax)
    for i, (o, r) in enumerate(rows):
        sm = np.array(r["samples"], float)
        ax.hlines(i, sm.min(), sm.max(), color=CYAN, lw=1.4, alpha=0.4, zorder=2)
        ax.plot(sm, [i] * len(sm), "o", ms=6.5, mfc=CYAN, mec="white", mew=1.0, zorder=3)
        ax.plot([r["value"]], [i], "D", ms=8, color=INK, zorder=4)
        if o in OLD_GRID:
            ax.plot([OLD_GRID[o]], [i], "X", ms=11, color=RED, zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NICE.get(o,o)}  (n={r['n']})" for o, r in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("friction μ measured from deceleration", color=MUTED, fontsize=9)
    ax.set_xlim(-0.03, 1.20)
    ax.plot([], [], "o", mfc=CYAN, mec="white", ms=6.5, label="individual take")
    ax.plot([], [], "D", color=INK, ms=7, label="combined measurement")
    ax.plot([], [], "X", color=RED, ms=9, label="retired grid fitter")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.02))
    return embed(fig)


def fig_audit(a):
    pk = a["per_kind"]
    kinds = ["drop", "slide", "collide"]
    fig, ax = plt.subplots(figsize=(8.4, 2.35))
    style(ax)
    y = np.arange(len(kinds)); left = np.zeros(len(kinds))
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


def fig_yield(y):
    labels = [f"{k}\n{v:.2f} m" for k, v in y["heights"]]
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    style(ax)
    x = np.arange(len(labels))
    pct = [100.0 * a / b for a, b in zip(y["ok"], y["tot"])]
    ax.bar(x, pct, color=CYAN, width=0.55)
    for i, (a, b) in enumerate(zip(y["ok"], y["tot"])):
        ax.text(i, pct[i] + 1.5, f"{a}/{b}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("% of drops yielding a value", color=MUTED, fontsize=8.5)
    ax.set_ylim(0, max(pct) * 1.35)
    return embed(fig)


def drop_yield(cfg):
    """Recompute yield per staged height from the cached tracks, so the figure is data."""
    from scripts.simple_fit import pixel_scale  # noqa: F401  (kept for parity)
    from src.motion.observables import admissible, drop_observables
    hs = cfg.get("heights", {"low": 0.08, "mid": 0.18, "high": 0.30})
    order = [k for k in ("low", "mid", "high") if k in hs]
    ok = {k: 0 for k in order}; tot = {k: 0 for k in order}
    # use the corrected, image-derived track cache when it exists; reading the stale
    # ptrk_ files here would have put pre-fix yields in a post-fix report
    tag = "_img" if (EXP / "seeds_image.json").exists() else ""
    for key, e in cfg["experiments"].items():
        if e["kind"] != "drop" or e.get("height_name") not in ok:
            continue
        h = e["height_name"]
        for f in sorted(EXP.glob(f"vid_{key}_seed*.npz")):
            c = EXP / f"ptrk{tag}_{f.stem.replace('vid_','')}_subject.npz"
            if not c.exists():
                continue
            d = np.load(c); tot[h] += 1
            if float(d["ncc_median"]) < 0.55 or float(d["ncc_end"]) < 0.55:
                continue
            o = drop_observables(np.stack([d["u"], d["v"]], 1))
            if o.get("ok") and admissible("restitution", o["value"]):
                ok[h] += 1
    return {"heights": [(k, hs[k]) for k in order],
            "ok": [ok[k] for k in order], "tot": [tot[k] for k in order]}


def main():
    d = json.loads((LAB / "simple_fit.json").read_text())
    a = json.loads((LAB / "audit_pixel.json").read_text())
    cfg = json.loads((EXP / "lab.json").read_text())
    fit = json.loads((EXP / "expand_fit.json").read_text())
    cmps = json.loads((DOCS / "comparisons.json").read_text())

    cc = a["counts"]; total = sum(cc.values())
    moves, static, degraded = cc["MOVES"], cc["STATIC"], cc["DEGRADED"]
    y = drop_yield(cfg)
    f_prec, f_two, f_aud, f_yield = (fig_precision(fit), fig_two_estimators(d),
                                     fig_audit(a), fig_yield(y))
    yld = 100.0 * sum(y["ok"]) / max(sum(y["tot"]), 1)

    tight = []
    for subj, r in fit.items():
        if "e_mid" in r and r["n_drop"] > 1 and r["e_mid"] > 0 \
                and r["e_mid_se"] / r["e_mid"] < 0.25:
            tight.append((f"{NICE.get(subj,subj)} restitution", r["e_mid"],
                          r["e_mid_se"]))
        c2 = r.get("friction")
        if c2 and c2["n"] > 1 and c2["value"] > 0 \
                and c2["interval"] / c2["value"] < 0.25:
            tight.append((f"{NICE.get(subj,subj)} friction", c2["value"],
                          c2["interval"]))
    tight.sort(key=lambda t: t[2] / t[1])

    trows = ""
    for subj in sorted(fit):
        r = fit[subj]
        em = ("—" if "e_mid" not in r
              else f'{r["e_mid"]:.3f} <span class="pm">± {r["e_mid_se"]:.3f}</span>')
        sl = ("—" if "slope" not in r
              else f'{r["slope"]:+.3f} <span class="pm">± {r["slope_se"]:.3f}</span>')
        fc = r.get("friction")
        fr = ("—" if not fc
              else f'{fc["value"]:.3f} <span class="pm">± {fc["interval"]:.3f}</span>')
        trows += (f'<tr><th>{NICE.get(subj,subj)}</th><td>{r["n_drop"]}</td>'
                  f'<td>{em}</td><td>{sl}</td><td>{fr}</td></tr>')

    vids = ""
    for k in ("ceramic_vase_drop_mid", "ceramic_vase_slide", "rubber_duck_drop_mid"):
        m = cmps.get(k)
        if not m:
            continue
        uri = embed_path(DOCS / m["file"], "image/webp")
        if not uri:
            continue
        vids += (f'<figure><img src="{uri}" alt="{k} generated versus simulated">'
                 f'<figcaption><b>{PRETTY_CMP.get(k, k)}</b> — {m["note"]}'
                 f'</figcaption></figure>')

    wins = "".join(f'<div class="win"><span>{nm}</span><b>{v:.3f} ± {e:.3f}'
                   f'&nbsp;&nbsp;({100*e/v:.0f}%)</b></div>' for nm, v, e in tight)

    html = f"""<title>Inverse Simulation from Video — Measurement Report</title>
<style>
:root {{ --ink:#141a22; --body:#33414f; --muted:#6b7a89; --line:#dde4ea;
  --bg:#ffffff; --card:#f7f9fb; --accent:#0e7c93; --warn:#b06d00; --bad:#b3261e;
  --good:#2f6b3a; --plate:#f7f9fb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e8eef4; --body:#b9c6d2; --muted:#8b9aa9; --line:#26313d;
    --bg:#0e141a; --card:#161f28; --accent:#4fc3d9; --warn:#e0a340; --bad:#f2836f;
    --good:#63b177; }}
}}
:root[data-theme="dark"] {{ --ink:#e8eef4; --body:#b9c6d2; --muted:#8b9aa9;
  --line:#26313d; --bg:#0e141a; --card:#161f28; --accent:#4fc3d9; --warn:#e0a340;
  --bad:#f2836f; --good:#63b177; }}
:root[data-theme="light"] {{ --ink:#141a22; --body:#33414f; --muted:#6b7a89;
  --line:#dde4ea; --bg:#ffffff; --card:#f7f9fb; --accent:#0e7c93; --warn:#b06d00;
  --bad:#b3261e; --good:#2f6b3a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--body);
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:62rem; margin:0 auto; padding:3.2rem 1.4rem 5rem; }}
.eyebrow {{ font:600 11.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent); }}
h1 {{ font-size:clamp(1.8rem,3.8vw,2.6rem); line-height:1.12; margin:.7rem 0 .5rem;
  color:var(--ink); letter-spacing:-.02em; text-wrap:balance; font-weight:650; }}
.lede {{ font-size:1.12rem; max-width:44rem; margin:0 0 2.2rem; }}
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
table {{ border-collapse:collapse; width:100%; min-width:36rem; font-size:.9rem; }}
th,td {{ text-align:left; padding:.6rem .8rem; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; }}
thead th {{ font:600 11.5px/1.3 ui-monospace,monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); }}
tbody th {{ font-weight:560; color:var(--ink); }}
td .pm {{ color:var(--muted); }}
.win {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:.8rem 1.1rem; margin:.5rem 0; display:flex; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; align-items:baseline; }}
.win b {{ color:var(--good); font-variant-numeric:tabular-nums; }}
.call {{ border-left:3px solid var(--accent); background:var(--card);
  padding:1rem 1.15rem; border-radius:0 9px 9px 0; margin:1.4rem 0; }}
.call.bad {{ border-left-color:var(--bad); }}
.call.good {{ border-left-color:var(--good); }}
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
<h1>What survives when the instrument is checked: two parameters, one reproducible
simulation, and a tracker that was aimed at the wrong place</h1>
<p class="lede">We stage physics experiments, film them with a video model, and recover
material parameters by measuring the motion. Everything below is measured. Every earlier
claim that did not survive re-measurement is listed at the end rather than removed.</p>

<div class="headline">
  <div><b>{len(tight)}</b><span>parameters measured to better than ±25% from more than
    one take — against zero before the expanded lab</span></div>
  <div><b>168</b><span>clips across 7 objects and 3 drop heights; {yld:.0f}% of drops
    yield a value</span></div>
  <div><b>{degraded} / {total}</b><span>earlier takes where the object stopped being the
    object — a failure no parameter can explain</span></div>
</div>

<h2><span class="num">01</span>What the pipeline recovers</h2>
{wins}
<div class="call bad"><b>An earlier version of this page reported three parameters,
including a wooden-bowl friction of 0.317 described as the first value to land in the
textbook range for wood on wood. That was an artefact and is withdrawn.</b>
<p>The tracking patch was being aimed by projected geometry that disagreed with the
renderer, and for four of seven objects it sat <em>off the object entirely</em>. Seeding
from the rendered image instead moves the bowl to 0.062 ± 0.151 — not a measurement — and
the rubber duck's restitution from 0.125 to 0.025 ± 0.010. Section 05 has the detail. Both
surviving parameters belong to the same object.</p></div>
<figure><img src="{f_prec}" alt="Every parameter with its interval">
<figcaption>Every parameter with its propagated interval, sorted by precision. No
handbook comparison is drawn: a downloaded mesh has no true density, and the goal is a
parameter that <em>explains the clip</em>, not one that matches a reference book.
</figcaption></figure>

<h2><span class="num">02</span>Generated video against our simulation of it</h2>
<p>Left pane: the clip Cosmos produced. Right pane: the simulated rollout at the parameter
recovered from that same clip, rendered by <b><code>newton.viewer.ViewerRTX</code> in the
staged scene</b> — the same table and objects, textured, ray traced, entirely inside
Warp/Newton with no Blender in the loop.</p>
<div class="call"><b>Physics and rendering are separate layers that meet at one interface:
per-frame object transforms.</b>
<p>Warp integrates a sphere-cover proxy (802 spheres for the vase) and knows nothing about
texture. The renderer needs textured triangles and a camera and knows nothing about contact.
They meet only at the per-frame transform, which is why the same rollout can be drawn by
ViewerRTX or by Blender without the physics changing. An earlier ViewerRTX pass looked bare
only because it had been handed a single mesh on a default ground plane;
<code>log_mesh</code> takes UVs, textures, roughness and metallic, so the staged scene can
be rebuilt inside it.</p></div>
<p>Two earlier versions of this pane are gone. The first faked it by cutting the object out
of a photograph with a mask and pasting it at the simulated position — every defect in it
was a mask failing to match the object's shape, and worse, it <em>concealed</em> the bug in
section 03 for the entire project by pasting a photo of an upright vase. The second was a
hand-written rasteriser, which was the wrong answer to the same question when a real
renderer was already available.</p>
{vids}
<div class="call"><b>What the vase render shows.</b>
<p>The simulated vase now starts upright — the axis fix — and then <b>topples on
landing</b>, while the generated clip keeps it standing. The rebound magnitudes agree
(measured 0.033, simulated 0.036); what disagrees is the post-impact behaviour. A tall
narrow vase dropped 18&nbsp;cm toppling is not obviously wrong physics, and Cosmos keeping
it perfectly upright is not obviously right. This is the first comparison in the project
where that question is even legible.</p></div>
<div class="call bad"><b>The rubber duck still does not reproduce: measured 0.025,
simulated 0.000.</b>
<p>The simulator produces no measurable rebound at all for that mesh at the recovered
damping. Worth stating plainly: this conclusion has now survived the axis fix, whereas the
earlier version of it (a floor at 0.194) was computed on a sideways mesh and meant
nothing.</p></div>

<h2><span class="num">03</span>The simulator was holding the objects on their sides</h2>
<p>glTF is Y-up by specification. Blender's importer converts to Z-up on load;
<code>trimesh</code> does not, and <code>load_asset</code> never applied the rotation. So
for the whole project the simulator held meshes in a different orientation from the
renderer:</p>
<div class="tw"><table>
<thead><tr><th>object</th><th>simulated footprint</th><th>simulated height</th>
<th>actual</th></tr></thead><tbody>
<tr><th>ceramic vase</th><td>12.6 × 26.5 cm</td><td>13.1 cm</td>
<td>12.6 × 12.6, 24.8 tall</td></tr>
<tr><th>wooden bowl</th><td>19.3 × 5.7 cm</td><td>19.1 cm</td>
<td>19.4 × 19.2, 5.8 tall</td></tr>
<tr><th>rubber duck</th><td>13.3 × 17.5 cm</td><td>18.4 cm</td><td>correct by luck</td></tr>
</tbody></table></div>
<p>The vase had been falling on its side and the bowl standing on its rim. Sphere covers,
resting heights and all contact geometry were built from the wrong pose, so every
simulation-side result predating this fix is void. The measurements are not affected —
restitution and friction are read from tracked pixel motion and never touch the mesh.</p>

<h2><span class="num">04</span>All seven objects</h2>
<p>Three drop heights turn restitution from one number into a relationship:
<code>e</code> at the centre of the measured speed range, and <code>de/dv</code>, its
change per m/s of impact speed. Impact speed is measured from the track, never assumed
from the staged height — the video model makes no promise to obey gravity.</p>
<div class="tw"><table>
<thead><tr><th>object</th><th>n drops</th><th>e at mid speed</th><th>de/dv</th>
<th>friction μ</th></tr></thead><tbody>{trows}</tbody></table></div>
<p>Speed dependence resolves for only two objects, and they disagree in sign: the duck's
<b>−0.849 ± 0.218</b> has the physically expected sign (restitution falls as impact speed
rises), the baseball's <b>+0.517 ± 0.080</b> does not. The book yielded nothing at all
from 18 drops — a flat slab tumbles, and tumbling lifts its own tracked centroid, which
reads as a rebound larger than the fall.</p>
<figure><img src="{f_yield}" alt="Drop yield by staged height">
<figcaption>Yield by staged drop height. Higher is <em>worse</em>: staging an object
further from its rendered context makes the model likelier to re-render it than to move
it. 0.18 m is the sweet spot.</figcaption></figure>

<h2><span class="num">05</span>The estimator was manufacturing numbers</h2>
<p>The retired fitter searched a parameter grid for the value whose hand-weighted motion
signature best matched the track. A grid always returns a best match; it never asks
whether the observation constrains anything. This comparison runs both estimators over
<em>identical</em> tracks.</p>
<figure><img src="{f_two}" alt="Per-take friction versus the grid fitter's answer">
<figcaption>The baseball's ten takes span 0.007–0.285. The retired grid fitter reported
<b>0.680</b> from those same ten takes — outside the range any individual take supports.
The apple's 1.100 was the grid maximum, a railed bound reported as a measurement.
</figcaption></figure>

<h2><span class="num">06</span>The tracker was aimed by the wrong pipeline</h2>
<p>Seeds were produced by projecting a body centre computed as <code>pos + vmean</code>,
the mesh's vertex mean added to its placed position, with the object's <code>rot_z</code>
never applied. Blender places the origin at <code>pos</code> and rotates it. The two
conventions disagree by the mesh's own centroid offset, which is near zero for a baseball
and large for anything asymmetric:</p>
<div class="tw"><table>
<thead><tr><th>object</th><th>seed u,v</th><th>true u,v</th><th>offset</th>
<th>patch half-size</th></tr></thead><tbody>
<tr><th>ceramic vase</th><td>186, 122</td><td>151, 114</td><td>+35, +8</td><td>28</td></tr>
<tr><th>rubber duck</th><td>183, 98</td><td>147, 118</td><td>+36, −20</td><td>29</td></tr>
<tr><th>brass pot</th><td>195, 108</td><td>155, 196</td><td>+40, −88</td><td>41</td></tr>
<tr><th>book</th><td>208, 135</td><td>181, 138</td><td>+27, −3</td><td>16</td></tr>
<tr><th>baseball</th><td>154, 135</td><td>152, 137</td><td>+1, −2</td><td>17</td></tr>
</tbody></table></div>
<p>Where the offset exceeds the patch half-size, the tracker was following mostly wall.
The fix is not a better projection but removing the parallel pipeline: the subject is the
only thing that moves between two stagings of the same object, so differencing two initial
frames isolates it exactly, <em>in the image we are about to track</em>. Geometry
conventions cannot disagree with the renderer if the renderer is the source. 18 of 28
seeds moved by more than 20&nbsp;px.</p>

<h2><span class="num">07</span>A withdrawn claim about gravity, and what a control showed</h2>
<p>An earlier version of this page stated that <del>the generated video runs about 5× too
slow, an effective gravity near 0.3&nbsp;m/s²</del>. That is withdrawn. It came from a
measurement that fails on a control where the answer is known.</p>
<p>The control is our own simulation, rendered by Cycles, where gravity <em>is</em>
9.81&nbsp;m/s² by construction. Running the same measurement on it returned
−0.31&nbsp;m/s². The fault was window selection: the code took the lowest point of the
trajectory, which for a vase that lands, rebounds and then topples is well after impact,
so the parabola was fitted across all three phases.</p>
<div class="tw"><table>
<thead><tr><th>measured from</th><th>window</th><th>g</th></tr></thead><tbody>
<tr><th>world z, no camera</th><td>6 samples (true free fall)</td><td>7.10 m/s²</td></tr>
<tr><th>image y, through the camera</th><td>6 samples</td><td><b>9.44 m/s²</b></td></tr>
<tr><th>either</th><td>7 samples (one frame past contact)</td><td>sign flips</td></tr>
</tbody></table></div>
<div class="call"><b>With the right window the pipeline recovers 9.44 against a true 9.81 —
4% — through the camera and the pixel scale.</b>
<p>So the physics, the projection and the px↔m scale are sound. What is not sound is
choosing the window: a 0.18&nbsp;m fall is five frames at 24&nbsp;fps, and one frame past
contact inverts the answer. Until that selection is robust, <b>the generated video's
gravity is unmeasured</b> — not wrong. Every per-clip figure produced by the old method is
void.</p></div>

<h2><span class="num">08</span>The audit that corrected itself</h2>
<p>An earlier version of this report stated that <del>68 of 93 takes contained no
measurable motion</del>. That came from tracked centroids, and the tracker fails on these
clips in a way its own diagnostics cannot see: points stay &ldquo;visible&rdquo; while
sitting on the background, so a duck that visibly fell to the table measured 6&nbsp;px of
motion. Re-run with appearance matching, {moves} of {total} takes move and only {static}
are genuinely static.</p>
<figure><img src="{f_aud}" alt="Audit of every take by probe">
<figcaption>{moves} move with the asset intact, {static} are static, {degraded} degrade.
</figcaption></figure>
<div class="call bad"><b>The brass pot stops being a brass pot.</b>
<p>All {degraded} degraded takes are the same object, which transforms mid-clip from a
lidded pot into a wide shallow bowl. No density, friction or restitution turns a pot into
a bowl, so this fails even a plausibility bar. It is asset integrity, not
identifiability.</p></div>

<h2><span class="num">09</span>Corrections</h2>
<ul>
<li><b>The simulator held meshes on their sides</b> for the whole project (glTF Y-up
never converted to Z-up), so every simulation-side result predating the fix is void.</li>
<li><b>Three published parameters were artefacts of off-object tracking.</b> The
wooden bowl's friction and the rubber duck's restitution do not survive; the vase's
restitution moved from 0.094 to 0.033.</li>
<li><b>&ldquo;The videos do not contain the experiment&rdquo; was an instrument
artefact</b>, stated twice on tracker evidence. {moves} of {total} takes move.</li>
<li><b>The published baseball restitution was 0.000 — no bounce at all.</b> On validated
tracks it is 0.215 ± 0.060; the point tracker was missing the rebound outright.</li>
<li><b>The noise floor was assumed, and assumed too low</b> (1.50&nbsp;px against a
measured 2.04&nbsp;px), so every earlier interval was about a third too narrow.</li>
<li><b>A claimed ~80% yield for drop and slide was wrong</b>; the real drop yield is
{yld:.0f}%. &ldquo;Moves&rdquo; and &ldquo;yields a usable observable&rdquo; are
different questions.</li>
<li><b>A max-excursion statistic is not a displacement.</b> Used twice — first as path
length, then again as peak displacement — and wrong both times.</li>
<li><b>60&nbsp;fps was not the diagnosis for mass</b> and is no longer proposed as a
requirement.</li>
<li><b>&ldquo;The generated video runs 5× too slow&rdquo; is withdrawn.</b> The measurement
returned −0.31&nbsp;m/s² on a control whose true gravity is 9.81. With the window corrected
the same pipeline reads 9.44.</li>
<li><b>There is no ground truth in this pipeline.</b> Errors were once quoted against
&ldquo;true&rdquo; densities from a hardcoded table of textbook guesses.</li>
</ul>

<h2><span class="num">10</span>Known limits</h2>
<ul>
<li>The rubber duck's values come from clips in which it re-forms rather than
translating; appearance matching cannot fully separate &ldquo;translated&rdquo; from
&ldquo;re-rendered nearby&rdquo;.</li>
<li>A prediction I could not test: I expected the baseball's backwards de/dv to flip with
more data. Its drop clips were already complete in the partial run, so the fit is over
identical takes and the prediction was never actually put to the test.</li>
<li>The physics proxy and the rendered mesh are different representations of the same
object — 802 spheres versus a textured glTF — connected only by a transform. That is normal,
but it is exactly where the Y-up/Z-up bug hid: nothing checked that the two agreed.</li>
<li>Collide was dropped from the expanded lab at ~24% yield, so mass ratio is not
measured here at all.</li>
</ul>

<footer>Generated by <code>scripts/make_report.py</code> from
<code>expand_fit.json</code>, <code>audit_pixel.json</code> and
<code>comparisons.json</code>. Full chronology, including every retraction, in
<code>docs/PROJECT_LOG.md</code>.</footer>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB)")
    print(f"  {len(tight)} parameters within ±25%; {len(cmps)} comparisons embedded; "
          f"drop yield {yld:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
