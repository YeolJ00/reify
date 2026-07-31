"""Put the generated clip and our simulation of it side by side.

There is no photoreal renderer in this project, so the "simulated video" is composited:
the object is inpainted out of the staged initial frame to leave a clean plate, and its
own sprite is then pasted wherever Newton says it is on each step. Every pixel of the
object is therefore real, and every position is simulated -- which is exactly the pairing
we want to look at, and it is honest about being a composite rather than a render.

The simulation is run at the parameter this pipeline recovered from that same clip, so
the question the picture answers is: does a valid Newton rollout at the recovered theta
reproduce the motion the video showed? Under the standing framing -- any plausible
parameter that explains the video will do -- that is the whole acceptance test.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/make_sim_videos.py    (warp env)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "expand"
DOCS = REPO / "docs"
SUBSTEPS, K_CONTACT, PITCH = 60, 2500.0, 0.020
NF = 49
FPS = 24.0


def sprite_and_plate(img, u, v, hw, hh):
    """Cut the object out of the staged frame and inpaint the hole.

    hw/hh are HALF-EXTENTS in pixels, derived from the object's real width AND height.
    Sizing both from seeds.json's `size_px` was wrong: that is the smallest horizontal
    extent, chosen so a tracker seeds inside a thin object, and it is far too small for a
    tall one. On the 12.6 x 24.8 cm vase it gave a 44 px mask for a 120 px tall object,
    so the inpaint erased only the middle -- leaving the vase's top and bottom hanging in
    mid-air while the pasted mid-band fragment fell like a shadow.
    """
    h, w = img.shape[:2]
    hw = int(max(hw, 10)); hh = int(max(hh, 10))
    x0, x1 = max(int(u) - hw, 0), min(int(u) + hw, w)
    y0, y1 = max(int(v) - hh, 0), min(int(v) + hh, h)
    sprite = img[y0:y1, x0:x1].copy()
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(u), int(v)), (int(hw * 1.12), int(hh * 1.12)),
                0, 0, 360, 255, -1)
    plate = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    # a soft-edged alpha so the paste does not show a hard square seam
    sh, sw = sprite.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    # elliptical falloff matching the crop's aspect, so a tall object is not clipped
    d = np.hypot((yy - sh / 2.0) / (sh / 2.0), (xx - sw / 2.0) / (sw / 2.0))
    alpha = np.clip(2.2 * (1.0 - d), 0.0, 1.0)[..., None]
    return sprite, alpha, plate


def paste(plate, sprite, alpha, u, v):
    out = plate.copy()
    h, w = out.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = int(round(u - sw / 2.0)), int(round(v - sh / 2.0))
    x1, y1 = x0 + sw, y0 + sh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = sw - max(0, x1 - w), sh - max(0, y1 - h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return out
    a = alpha[sy0:sy1, sx0:sx1]
    out[y0:y1, x0:x1] = (a * sprite[sy0:sy1, sx0:sx1]
                         + (1 - a) * out[y0:y1, x0:x1]).astype(np.uint8)
    return out


def rollout(cfg, e_key, cd, mu, load_asset, decimate, sphere_cover, ProbeScene, Camera,
            v0=0.0):
    from src.lab.staging import obj_geom
    e = cfg["experiments"][e_key]
    so = cfg["assets"][e["subject"]]
    GZ = cfg["ground_z"]
    c, mk, ms = obj_geom(so, e["subject_pos"], load_asset, decimate, sphere_cover, GZ)
    c[2] += e["lift"]
    # A slide with zero initial velocity simply sits there: the launch speed is part of
    # the observation, not of the material, so it is measured from the clip and imposed.
    s = ProbeScene([mk], [list(c)], [[float(v0), 0.0, 0.0]], densities=(600.0,),
                   ground_z=GZ,
                   dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS, k=K_CONTACT,
                   cd=float(cd), mu=float(mu), mesh_scale=[ms], pitch=PITCH)
    s.rollout()
    P = s.positions(SUBSTEPS)[:NF]
    if not np.isfinite(P).all():
        return None
    cam = Camera(cfg["camera"])
    uv, _ = cam.project(P[:, 0])
    return uv


def tune_cd(target_e, *args):
    """Find the contact damping whose rollout rebounds like the measurement."""
    from src.motion.observables import drop_observables
    lo, hi, best = 0.5, 2000.0, None
    for _ in range(12):
        cd = float(np.sqrt(lo * hi))
        uv = rollout(*args[:1], args[1], cd, *args[2:])
        if uv is None:
            break
        # observables.drop_observables carries no observation-only plausibility gates
        # (those lived in the retired signature module), so a rollout can be measured
        # with exactly the same function used on the video -- which is the point
        o = drop_observables(np.stack([uv[:, 0], uv[:, 1]], 1))
        if not o.get("ok"):
            # A rollout with no detectable rebound is e ~ 0, not a reason to abandon the
            # search. Breaking here made the bisection return whatever it had reached so
            # far -- it reported cd=32/e=0.001 for a target of 0.033 while cd=4 gave
            # 0.094, so the answer was bracketed the whole time and simply never found.
            hi = cd
            continue
        # keep the CLOSEST rollout, not the last one the bisection happened to try:
        # if the target lies outside what the simulator can produce, the last iterate
        # is an arbitrary endpoint while the closest is the honest best effort
        if best is None or abs(o["value"] - target_e) < abs(best[1] - target_e):
            best = (cd, o["value"], uv)
        if o["value"] > target_e:
            lo = cd
        else:
            hi = cd
    return best


def main():
    import warp as wp

    from src.data.assets import decimate, load_asset
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover
    from src.sim.probe_scene import ProbeScene

    cfg = json.loads((LAB / "lab.json").read_text())
    si = LAB / "seeds_image.json"
    seeds = json.loads((si if si.exists() else LAB / "seeds.json").read_text())
    print(f"seeds: {'image-derived' if si.exists() else 'projected'}")
    fit = json.loads((LAB / "expand_fit.json").read_text())

    # the objects whose parameters actually cleared the precision bar, plus the height
    # whose clips produced them
    # the parameters that survive on corrected seeds, plus one that does not
    picks = [("ceramic_vase", "drop_mid", "restitution"),
             ("ceramic_vase", "slide", "friction"),
             ("rubber_duck", "drop_mid", "restitution")]

    wp.init()
    out = {}
    with wp.ScopedDevice("cuda:0"):
        for subj, stage, what in picks:
            key = f"{subj}_{stage}"
            if key not in cfg["experiments"]:
                print(f"  {key}: not staged"); continue
            r = fit.get(subj, {})
            mu = (r.get("friction") or {}).get("value") or 0.4
            e_t = r.get("e_mid")
            # Pick a take that ACTUALLY PRODUCED the measurement, then the best-tracked
            # among those. Selecting purely on appearance match chose a clip whose slide
            # observable was empty, so the simulation was launched at 0 m/s and sat still
            # -- comparing our simulation against a take that contributed nothing.
            from scripts.simple_fit import pixel_scale as _ps
            from src.motion.observables import (drop_observables as _do,
                                                slide_observables as _so)
            _ppm, _ = _ps(cfg["camera"], cfg["ground_z"])
            best = None
            for f in sorted(LAB.glob(f"vid_{key}_seed*.npz")):
                c = LAB / f"ptrk_img_{f.stem.replace('vid_','')}_subject.npz"
                if not c.exists():
                    continue
                d = np.load(c)
                cen = np.stack([d["u"], d["v"]], 1)
                o = (_do(cen) if what == "restitution" else _so(cen, _ppm, FPS))
                if not o.get("ok"):
                    continue
                if best is None or float(d["ncc_median"]) > best[1]:
                    best = (f, float(d["ncc_median"]))
            if best is None:
                print(f"  {key}: no take yielded an observable"); continue
            if best is None:
                print(f"  {key}: no tracked take"); continue
            clip, ncc = best
            frames = np.load(clip)["frames"]

            if what == "restitution" and e_t is not None:
                got = tune_cd(e_t, cfg, key, mu, load_asset, decimate, sphere_cover,
                              ProbeScene, Camera)
                if got is None:
                    print(f"  {key}: rollout failed"); continue
                cd, e_sim, uv = got
                note = f"e measured {e_t:.3f}, simulated {e_sim:.3f}, cd={cd:.0f}"
            else:
                cd = 20.0
                # observed launch speed of THIS take, px/frame -> m/s
                from scripts.simple_fit import pixel_scale
                from src.motion.observables import slide_observables
                ppm, _ = pixel_scale(cfg["camera"], cfg["ground_z"])
                ck = LAB / f"ptrk_img_{clip.stem.replace('vid_','')}_subject.npz"
                dtr = np.load(ck)
                so_ = slide_observables(np.stack([dtr["u"], dtr["v"]], 1), ppm, FPS)
                v0 = abs(so_.get("v0", 0.0)) * FPS / ppm if so_.get("ok") else 0.0
                uv = rollout(cfg, key, cd, mu, load_asset, decimate, sphere_cover,
                             ProbeScene, Camera, v0=v0)
                if uv is None:
                    print(f"  {key}: rollout failed"); continue
                note = f"mu measured {mu:.3f}, launched at {v0:.2f} m/s (from the clip)"

            s = seeds[key]["subject"]
            I0 = frames[0]
            # projected half-extents from the asset's real dimensions
            # extents measured from the image too, when available: deriving them from
            # asset geometry is what put the mask beside the object in the first place
            if "w_px" in s:
                hw, hh = 0.5 * s["w_px"], 0.5 * s["h_px"]
            else:
                so_cm = cfg["assets"][subj]["size_cm"]
                ppm_obj = s["size_px"] / (min(so_cm[0], so_cm[1]) / 100.0)
                hw = 0.5 * ppm_obj * max(so_cm[0], so_cm[1]) / 100.0
                hh = 0.5 * ppm_obj * so_cm[2] / 100.0
            sprite, alpha, plate = sprite_and_plate(I0, s["u"], s["v"], hw, hh)
            # Anchor the simulated path to the observed starting point. obj_geom centres
            # a body on its sphere-cover origin while the tracker seeds on the projected
            # centroid, so the two differ by a constant offset; leaving it in made the
            # sprite land beside the hole it was cut from and read as a smudge. The
            # comparison is about MOTION, and a constant offset carries no dynamics.
            du = s["u"] - float(uv[0, 0]); dv = s["v"] - float(uv[0, 1])
            sim = np.stack([paste(plate, sprite, alpha,
                                  uv[t, 0] + du, uv[t, 1] + dv)
                            for t in range(min(NF, len(uv)))])
            n = min(len(sim), len(frames))
            np.savez_compressed(LAB / f"cmp_{key}.npz", generated=frames[:n],
                                simulated=sim[:n], note=note)
            out[key] = {"note": note, "clip": clip.name, "ncc": ncc, "n": int(n)}
            print(f"  {key}: {note}  (vs {clip.name}, ncc {ncc:.2f})")

    (LAB / "sim_videos.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} comparisons to {LAB}/cmp_*.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
