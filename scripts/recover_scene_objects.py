"""Recover each scene object's restitution from its own generated drop-probe video.

Per object:
  1. seed CoTracker inside the object's projected silhouette in frame 0 (no colour
     assumptions — these are arbitrary props, not the coloured balls);
  2. screen the take for physical validity (does it actually fall? is it trackable?
     does the path bend with nothing pushing it?);
  3. fit a rigid drop through the differentiable-contact sim, projected through the
     scene camera, with restitution the unknown and the launch velocity a nuisance;
  4. report the value AND whether it was actually identified (effect above the noise
     floor) — an unidentified object keeps a class prior and is flagged.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/recover_scene_objects.py    (warp env)
Stage 1 (tracking) needs the `video` env; it is cached to disk so this script can be
run in two passes: TRACK_ONLY=1 with the video env, then the fit with the warp env.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "scene" / "probes"
NOISE_PX = 2.5
CD_RANGE = (2.0, 400.0)   # normal damping: low = bouncy, high = dead thud
# If the simulator cannot reproduce the generated motion AT ALL, the parameter that
# best fits garbage is still garbage. A large spread across the scan is not evidence
# of identification unless the fit itself is credible.
MAX_FIT_PX = 25.0
K_CONTACT, SUBSTEPS = 2500.0, 80


def object_silhouette(frames, cam, drop_pos, size_m):
    """Rough mask of the target object in frame 0 from its projected position + size."""
    import numpy as np
    uv, dep = cam.project(np.array([drop_pos]))
    u, v = uv[0]
    r = 0.42 * cam.fx * size_m / max(float(dep[0]), 1e-6)
    H, W, _ = frames[0].shape
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - u) ** 2 + (yy - v) ** 2) < r ** 2, (u, v), r


def write_seeds():
    """Stage 0 (warp env) — where each object appears in frame 0, for the tracker.
    Kept separate because src.render.camera imports warp, which the video env lacks."""
    from src.render.camera import Camera
    cfg = json.loads((OUT / "probes.json").read_text())
    scene_objs = json.loads((REPO / "outputs" / "scene" / "scene.json").read_text())["objects"]
    cam = Camera(cfg["camera"])
    seeds = {}
    for name, p in cfg["probes"].items():
        so = scene_objs[name]
        rc = 0.5 * (np.array(so["bbox_min"]) + np.array(so["bbox_max"]))
        drop_c = list(rc + np.array([0, 0, p["drop_h"]]))
        uv, dep = cam.project(np.array([drop_c]))
        # radius from the SMALLEST horizontal extent, not the largest dimension: a tall
        # thin object (the vase is 12.6 cm wide and 24.8 cm tall) otherwise gets a seed
        # circle full of background wall, and the tracker follows the wall instead.
        size_m = min(p["size_cm"][0], p["size_cm"][1]) / 100.0
        seeds[name] = {"u": float(uv[0][0]), "v": float(uv[0][1]),
                       "r": float(0.40 * cam.fx * size_m / max(float(dep[0]), 1e-6))}
        print(f"  seed {name:13s} at ({seeds[name]['u']:.0f},{seeds[name]['v']:.0f}) r={seeds[name]['r']:.0f}px")
    (OUT / "seeds.json").write_text(json.dumps(seeds, indent=2))
    print("wrote", OUT / "seeds.json")
    return 0


def track_objects():
    """Stage 1 — CoTracker on each probe video (run in the `video` env)."""
    import glob
    from src.track.cotracker import seed_in_mask, track_points
    cfg = json.loads((OUT / "probes.json").read_text())
    seeds = json.loads((OUT / "seeds.json").read_text())
    for name, p in cfg["probes"].items():
        sd = seeds[name]
        for f in sorted(glob.glob(str(OUT / f"vid_{name}_seed*.npz"))):
            seed = Path(f).stem.split("seed")[1]
            dst = OUT / f"trk_{name}_seed{seed}.npz"
            if dst.exists():
                continue
            fr = np.load(f)["frames"]
            H, W, _ = fr[0].shape
            yy, xx = np.mgrid[0:H, 0:W]
            u, v, r = sd["u"], sd["v"], sd["r"]
            mask = ((xx - u) ** 2 + (yy - v) ** 2) < r ** 2
            q = seed_in_mask(mask, n=40, seed=0)
            tracks, vis = track_points(fr, q, device="cuda")
            cen = np.array([np.median(tracks[t, vis[t]] if vis[t].mean() > 0.3 else tracks[t], axis=0)
                            for t in range(len(fr))])
            np.savez(dst, tracks=tracks, vis=vis, cen=cen, seed_uv=[u, v], seed_r=r)
            print(f"tracked {name} seed{seed}: {vis.mean()*100:.0f}% visible", flush=True)
    return 0


def screen(cen, cam, drop_pos, rest_pos):
    """Physical-validity gates. Returns (ok, reasons, fall_px)."""
    bad = []
    m = ~np.isnan(cen[:, 0])
    if m.mean() < 0.6:
        bad.append(f"tracked only {m.mean()*100:.0f}% of frames")
    P = cen[m]
    uv_d, _ = cam.project(np.array([drop_pos])); uv_r, _ = cam.project(np.array([rest_pos]))
    expect = float(uv_r[0][1] - uv_d[0][1])            # how far down it should land, px
    fell = float(P[-1][1] - P[0][1])
    if expect > 4 and fell < 0.35 * expect:
        bad.append(f"barely falls ({fell:.0f}px vs {expect:.0f}px expected)")
    # falling much FURTHER than the drop height allows means the object left the table
    # or the tracker slid off it — either way the clip is not the experiment we asked for
    if expect > 4 and fell > 1.7 * expect:
        bad.append(f"falls {fell:.0f}px, far past the {expect:.0f}px it was lifted")
    d = P[-1] - P[0]; L = float(np.linalg.norm(d))
    if L > 12:
        n = np.array([-d[1], d[0]]) / L
        if float(np.abs((P - P[0]) @ n).max()) / L > 0.18:
            bad.append("path curves with nothing pushing it")
    return len(bad) == 0, bad, fell, expect


def main():
    if os.environ.get("SEED_ONLY"):
        return write_seeds()
    if os.environ.get("TRACK_ONLY"):
        return track_objects()

    import warp as wp
    from src.render.camera import Camera
    from src.sim.probe_scene import ProbeScene
    from src.data.assets import load_asset

    cfg = json.loads((OUT / "probes.json").read_text())
    scene_objs = json.loads((REPO / "outputs" / "scene" / "scene.json").read_text())["objects"]
    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]
    results = {}
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        for name, p in cfg["probes"].items():
            trks = sorted(OUT.glob(f"trk_{name}_seed*.npz"))
            if not trks:
                print(f"{name}: no tracks yet"); continue
            best = None
            for t in trks:
                d = np.load(t); cen = d["cen"]
                so = scene_objs[name]
                rc = 0.5 * (np.array(so["bbox_min"]) + np.array(so["bbox_max"]))
                ok, why, fell, expect = screen(cen, cam, list(rc + [0, 0, p["drop_h"]]), list(rc))
                seed = t.stem.split("seed")[1]
                print(f"  {name:13s} seed{seed}: {'usable' if ok else 'REJECT ' + '; '.join(why)}"
                      f" (falls {fell:.0f}px)")
                # choose the take whose fall best MATCHES the drop we staged, not the
                # biggest one — the biggest is usually the most anomalous clip
                if ok and (best is None or abs(fell - expect) < abs(best[1] - expect)):
                    best = (cen, fell, seed)
            if best is None:
                results[name] = {"identified": False, "reason": "no usable take"}
                continue
            cen, fell, seed = best
            # Use the object's REAL geometry (sphere-covered, scaled as placed in the
            # scene) rather than an equivalent sphere: a vase and a duck do not land like
            # a ball. The body origin is the mesh's vertex mean, so we fit RELATIVE motion
            # (displacement from frame 0) — that cancels any constant offset between the
            # sim's origin and the visual centroid the tracker follows.
            so = scene_objs[name]
            asset_name = Path(so["asset"]).name
            cat = so["asset"].split("/")[0]
            tm = load_asset("rigid" if "rigid" in so["asset"] else "soft", asset_name)
            vmean = np.asarray(tm.vertices).mean(0) * so["scale"]
            rest_c = np.array(so["pos"], float) + vmean
            drop_c = [float(rest_c[0]), float(rest_c[1]), float(rest_c[2] + p["drop_h"])]
            nf = len(cen)
            m = ~np.isnan(cen[:, 0])
            i0 = int(np.argmax(m))              # first tracked frame — the SAME reference
            obs_rel = cen - cen[i0]             # must be used for sim and observation

            def rms(cd, v0):
                # a mesh cover applies contact through HUNDREDS of spheres at once, so the
                # per-sphere stiffness must be far lower than for a single-sphere ball or
                # the body is launched off the table
                sc = ProbeScene([f'{cat}/{asset_name}'], [drop_c], [[v0[0], v0[1], 0.0]],
                                densities=(600.0,), ground_z=GZ, dt=1.0 / (24 * SUBSTEPS),
                                n_steps=nf * SUBSTEPS, k=K_CONTACT, cd=float(cd), mu=0.4,
                                mesh_scale=[so["scale"]], pitch=0.020)
                sc.rollout(); P = sc.positions(SUBSTEPS)
                if not np.isfinite(P).all():
                    return 1e6
                uv, _ = cam.project(P[:nf, 0])
                sim_rel = uv - uv[i0]
                return float(np.sqrt(((sim_rel[m] - obs_rel[m]) ** 2).mean()))

            scan = np.geomspace(*CD_RANGE, 9)
            errs = []
            for cd in scan:
                b = 1e9
                for vx in np.linspace(-0.25, 0.25, 3):
                    for vy in np.linspace(-0.25, 0.25, 3):
                        b = min(b, rms(cd, (vx, vy)))
                errs.append(b)
            errs = np.array(errs); i = int(np.argmin(errs))
            spread = float(errs.max() - errs.min())
            railed = i in (0, len(scan) - 1)
            explains = float(errs[i]) <= MAX_FIT_PX
            ident = explains and spread > NOISE_PX and not railed
            why = ("ok" if ident else
                   ("sim cannot reproduce this motion" if not explains else
                    ("optimum outside the tested range" if railed else "parameter barely affects the motion")))
            results[name] = {"identified": bool(ident), "why": why,
                             "cd": float(scan[i]), "seed": seed,
                             "spread_px": spread, "fit_px": float(errs[i]), "railed": bool(railed),
                             "scan": scan.tolist(), "errs": errs.tolist()}
            print(f"  -> {name}: cd={scan[i]:.1f} fit={errs[i]:.1f}px spread={spread:.1f}px "
                  f"{'IDENTIFIED' if ident else 'NOT identified: ' + why}")

    (OUT / "recovered.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "recovered.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
