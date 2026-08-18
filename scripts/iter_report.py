"""Walk a fit's iteration history and emit ONE PAGE PER OBJECT.

A joint fit already writes everything needed to see what it did -- `results.json` holds the
per-object theta at every step, and `iters/it*/` holds the clips that produced it -- but
nothing joins them, so you cannot watch an object's parameters move and see the motion that
moved them. This does that join.

The unit is the OBJECT, not the iteration, because the fit is per-object: every object carries
its own theta vector and its own SPSA perturbation, and the only thing they share is the scene
they are simulated and rendered in. A page per iteration would mix objects that are being
optimised independently; a page per object shows one parameter trajectory against the probes
that drove it.

Each page shows, per iteration:
  * theta and how far each component moved since the previous step
  * the aggregate objective difference dy that SPSA descended
  * per-probe scores for the plus and minus perturbations, which is where dy comes from
  * the clips themselves, both perturbations side by side, for every probe and view

Run:  python scripts/iter_report.py [RUN_DIR] [--out DIR] [--max-clips N]
"""
import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
THETA_KEYS = ("mu", "cd", "rho")


def find_ffmpeg():
    """imageio_ffmpeg ships a binary and knows where it is -- ask it rather than guessing a
    path, which is version- and env-specific and silently produced clip-less pages."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def gif_data_uri(mp4, ff, width=300, fps=10):
    """mp4 -> an embedded GIF. Artifacts cannot fetch files, so clips must be inlined.

    Two-pass palette: a 24-bit render quantised straight to 256 colours bands badly on the
    smooth table and HDRI gradients that fill most of these frames.
    """
    if ff is None or not Path(mp4).exists():
        return None
    pal = Path("/tmp") / f"_pal_{abs(hash(str(mp4))) % 10**9}.png"
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    try:
        subprocess.run([ff, "-y", "-i", str(mp4), "-vf", f"{vf},palettegen=max_colors=64",
                        "-f", "image2", str(pal)], check=True, capture_output=True)
        out = subprocess.run(
            [ff, "-y", "-i", str(mp4), "-i", str(pal), "-lavfi",
             f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
             "-loop", "0", "-f", "gif", "-"],
            check=True, capture_output=True).stdout
    except Exception:
        return None
    finally:
        pal.unlink(missing_ok=True)
    return "data:image/gif;base64," + base64.b64encode(out).decode()


def load_history(run):
    res = json.loads((run / "results.json").read_text())
    hist, final = res.get("history", []), res.get("theta", {})
    objects = sorted({o for e in hist for o in e.get("objects", {})} | set(final))
    return hist, final, objects


def clips_for(run, k, obj, max_clips):
    """Clips from iteration k that show this object. Falls back to the top-level dir for the
    final state, which the fit leaves in place rather than copying into an iters/ folder."""
    d = run / "iters" / f"it{k:02d}"
    if not d.is_dir():
        d = run
    found = sorted(p for p in d.glob(f"*__{obj}@*.mp4"))
    # group by (probe, view) so the plus/minus pair sits together -- the pair IS the gradient
    pairs = {}
    for p in found:
        stem = p.stem
        tag = stem.split("_", 1)[0]
        rest = stem.split("_", 1)[1] if "_" in stem else stem
        probe = rest.split("__")[0]
        view = stem.rsplit("@", 1)[-1]
        pairs.setdefault((probe, view), {})[tag] = p
    # Spread the budget ACROSS PROBES, one view each, before spending a second slot on any
    # probe. Taking the first N alphabetically gave every slot to `collide` and the page never
    # showed the tilt or drop that the other theta components came from -- which defeats the
    # point, since the whole reason for a joint fit is that different probes constrain
    # different parameters.
    if not max_clips:
        return [(pr, vw, pairs[(pr, vw)]) for pr, vw in sorted(pairs)]
    by_probe = {}
    for pr, vw in sorted(pairs):
        by_probe.setdefault(pr, []).append(vw)
    picked, rnd = [], 0
    while len(picked) < max_clips and any(len(v) > rnd for v in by_probe.values()):
        for pr in sorted(by_probe):
            if len(by_probe[pr]) > rnd and len(picked) < max_clips:
                picked.append((pr, by_probe[pr][rnd]))
        rnd += 1
    return [(pr, vw, pairs[(pr, vw)]) for pr, vw in picked]


def fmt_theta(t):
    return " · ".join(f"{k}={t[k]:.4g}" for k in THETA_KEYS if k in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="outputs/judge/joint")
    ap.add_argument("--out", default="docs/iters")
    ap.add_argument("--max-clips", type=int, default=4,
                    help="probe x view pairs to embed per iteration (0 = all)")
    ap.add_argument("--width", type=int, default=300)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    if not (run / "results.json").exists():
        sys.exit(f"no results.json in {run}")
    out.mkdir(parents=True, exist_ok=True)
    ff = find_ffmpeg()
    if ff is None:
        print("  ! ffmpeg not found -- pages will have tables but no clips")
    hist, final, objects = load_history(run)
    print(f"  {len(hist)} iterations, {len(objects)} objects: {', '.join(objects)}")

    index = []
    for obj in objects:
        blocks, prev = [], None
        for e in hist:
            k = e.get("iter", 0)
            d = e.get("objects", {}).get(obj)
            if d is None:
                continue
            th = d.get("theta", {})
            delta = ("" if prev is None else " · ".join(
                f"{key} {(th[key] - prev[key]) / prev[key] * 100:+.1f}%"
                for key in THETA_KEYS if key in th and prev.get(key)))
            prev = th
            pp, pm = d.get("per_probe_plus", {}), d.get("per_probe_minus", {})
            probes = sorted(set(pp) | set(pm))
            prow = "".join(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td class='%s'>%s</td></tr>" % (
                    html.escape(p),
                    f"{pp[p]:+.3f}" if p in pp else "&mdash;",
                    f"{pm[p]:+.3f}" if p in pm else "&mdash;",
                    ("up" if (pp.get(p, 0) - pm.get(p, 0)) > 0 else "dn"),
                    f"{pp.get(p, 0) - pm.get(p, 0):+.3f}") for p in probes)
            cl = ""
            for probe, view, tags in clips_for(run, k, obj, a.max_clips):
                cells = ""
                for tag in ("plus", "minus"):
                    if tag not in tags:
                        continue
                    uri = gif_data_uri(tags[tag], ff, a.width)
                    cells += ("<figure><img src='%s' alt='%s %s'>"
                              "<figcaption>%s</figcaption></figure>" %
                              (uri, probe, tag, tag)) if uri else ""
                if cells:
                    cl += f"<div class='clipset'><h4>{html.escape(probe)} &middot; view {view}</h4><div class='pair'>{cells}</div></div>"
            dy = d.get("dy")
            blocks.append(f"""<section class="it">
  <div class="ith"><h3>iteration {k}</h3>
    <span class="dy">{'dy ' + format(dy, '+.3f') if dy is not None else ''}</span></div>
  <p class="th"><code>{html.escape(fmt_theta(th))}</code></p>
  {f'<p class="dl">{html.escape(delta)}</p>' if delta else ''}
  <div class="scroll"><table><thead><tr><th>probe</th><th>plus</th><th>minus</th>
    <th>difference</th></tr></thead><tbody>{prow}</tbody></table></div>
  {cl}
</section>""")
        page = TPL.format(obj=html.escape(obj), n=len(blocks),
                          final=html.escape(fmt_theta(final.get(obj, {}))),
                          body="\n".join(blocks))
        f = out / f"{obj}.html"
        f.write_text(page)
        index.append((obj, f.name, len(blocks)))
        print(f"  wrote {f}  ({len(blocks)} iterations)")

    rows = "".join(f"<li><a href='{n}'>{html.escape(o)}</a> &mdash; {c} iterations</li>"
                   for o, n, c in index)
    (out / "index.html").write_text(
        f"<title>Fit history</title><h1>Fit history</h1><ul>{rows}</ul>")
    print(f"  wrote {out}/index.html")


TPL = """<title>{obj} — fit history</title>
<style>
 :root{{--bg:#f5f2ed;--ink:#1b1a18;--soft:#615c55;--rule:#ddd6c9;--card:#fffdfa;
   --up:#2b6a4b;--dn:#a13c22;--m:ui-monospace,Menlo,Consolas,monospace;
   --d:ui-serif,Georgia,serif;--s:system-ui,-apple-system,sans-serif;}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#16181b;--ink:#e8e4dc;--soft:#9aa0a7;
   --rule:#2c3238;--card:#1d2126;--up:#62c193;--dn:#e08163;}}}}
 :root[data-theme="dark"]{{--bg:#16181b;--ink:#e8e4dc;--soft:#9aa0a7;--rule:#2c3238;
   --card:#1d2126;--up:#62c193;--dn:#e08163;}}
 :root[data-theme="light"]{{--bg:#f5f2ed;--ink:#1b1a18;--soft:#615c55;--rule:#ddd6c9;
   --card:#fffdfa;--up:#2b6a4b;--dn:#a13c22;}}
 body{{margin:0;padding:0 1.2rem 4rem;background:var(--bg);color:var(--ink);
   font-family:var(--s);line-height:1.6;}}
 .w{{max-width:54rem;margin:0 auto;}}
 header{{padding:3rem 0 1.4rem;}}
 h1{{font-family:var(--d);font-size:2.1rem;font-weight:600;margin:0 0 .3rem;
   letter-spacing:-.01em;}}
 .sub{{color:var(--soft);margin:0;}}
 .it{{border-top:1px solid var(--rule);padding:1.4rem 0;}}
 .ith{{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;}}
 h3{{font-family:var(--d);font-size:1.15rem;margin:0;}}
 h4{{font-size:.83rem;margin:0 0 .4rem;color:var(--soft);font-weight:500;}}
 .dy{{font-family:var(--m);font-size:.85rem;color:var(--soft);}}
 .th code{{font-family:var(--m);font-size:.88rem;background:color-mix(in srgb,var(--rule) 32%,transparent);
   padding:.15em .4em;border-radius:2px;}}
 .dl{{font-family:var(--m);font-size:.78rem;color:var(--soft);margin:.1rem 0 .7rem;}}
 .scroll{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;
   background:var(--card);margin-bottom:.9rem;}}
 table{{border-collapse:collapse;width:100%;font-family:var(--m);font-size:.79rem;
   font-variant-numeric:tabular-nums;white-space:nowrap;}}
 th,td{{padding:.35rem .7rem;text-align:right;border-bottom:1px solid var(--rule);}}
 th{{font-size:.66rem;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);
   font-weight:500;background:color-mix(in srgb,var(--rule) 22%,transparent);}}
 td:first-child,th:first-child{{text-align:left;}}
 tbody tr:last-child td{{border-bottom:none;}}
 .up{{color:var(--up);}} .dn{{color:var(--dn);}}
 .clipset{{margin:.8rem 0;}}
 .pair{{display:grid;grid-template-columns:1fr;gap:.6rem;}}
 @media(min-width:34rem){{.pair{{grid-template-columns:1fr 1fr;}}}}
 figure{{margin:0;border:1px solid var(--rule);border-radius:4px;overflow:hidden;
   background:var(--card);}}
 figure img{{display:block;width:100%;height:auto;}}
 figcaption{{padding:.3rem .6rem;font-size:.74rem;color:var(--soft);
   border-top:1px solid var(--rule);font-family:var(--m);}}
</style>
<div class="w">
<header>
 <h1>{obj}</h1>
 <p class="sub">{n} iterations · final <code>{final}</code></p>
 <p class="sub" style="font-size:.88rem;margin-top:.5rem">Each object carries its own θ and its
 own SPSA perturbation; only the scene is shared. The plus/minus clip pair below each step is
 the gradient — the difference between them is what moved θ.</p>
</header>
{body}
</div>"""


if __name__ == "__main__":
    main()
