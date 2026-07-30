"""Produce the data the fit reads: tracker seeds, then point tracks.

Two steps, in two different conda envs, which is why they are separate commands:

    seeds   (warp env)   projects each staged body to a seed disc -> seeds.json
    track   (video env)  runs CoTracker on every clip            -> trk_*.npz

Extracted unchanged from the retired grid fitter; only the estimator was replaced.

    CUDA_VISIBLE_DEVICES=<g> /path/envs/warp/bin/python  scripts/prepare_lab.py seeds
    CUDA_VISIBLE_DEVICES=<g> /path/envs/video/bin/python scripts/prepare_lab.py track
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# LAB=<dir> retargets both steps at another lab (e.g. the generator screen), so the
# seeding and tracking code is shared rather than copied per experiment set
LAB = Path(os.environ.get("LAB") or (REPO / "outputs" / "scene" / "fulllab"))


def write_seeds():
    from src.data.assets import decimate, load_asset
    from src.lab.staging import obj_geom
    from src.render.camera import Camera
    from src.sim.diff_collide_mesh import sphere_cover

    cfg = json.loads((LAB / "lab.json").read_text())
    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]
    out = {}
    for key, e in cfg["experiments"].items():
        s = {}
        for role, nm, pos in (("subject", e["subject"], e["subject_pos"]),
                              ("partner", e["partner"], e["partner_pos"])):
            if nm is None:
                continue
            so = cfg["assets"][nm]
            c, _k, _sc = obj_geom(so, pos, load_asset, decimate, sphere_cover, GZ)
            if role == "subject":
                c[2] += e["lift"]
            uv, dep = cam.project(np.array([c]))
            # the SMALLEST horizontal extent, not the largest: seeding a tall thin
            # vase from its largest dimension put tracker points on the wall behind it
            size_m = min(so["size_cm"][0], so["size_cm"][1]) / 100.0
            s[role] = {"u": float(uv[0][0]), "v": float(uv[0][1]),
                       "r": float(0.40 * cam.fx * size_m / max(float(dep[0]), 1e-6)),
                       "size_px": float(cam.fx * size_m / max(float(dep[0]), 1e-6))}
        out[key] = s
    (LAB / "seeds.json").write_text(json.dumps(out, indent=2))
    print(f"wrote seeds for {len(out)} experiments")
    return 0


def track():
    from src.track.cotracker import seed_in_mask, track_points

    seeds = json.loads((LAB / "seeds.json").read_text())
    for key, s in seeds.items():
        for f in sorted(glob.glob(str(LAB / f"vid_{key}_seed*.npz"))):
            sd = Path(f).stem.split("seed")[1]
            dst = LAB / f"trk_{key}_seed{sd}.npz"
            if dst.exists():
                continue
            fr = np.load(f)["frames"]; H, W, _ = fr[0].shape
            yy, xx = np.mgrid[0:H, 0:W]
            out = {}
            for role in ("subject", "partner"):
                if role not in s:
                    continue
                q0 = s[role]
                mask = ((xx - q0["u"]) ** 2 + (yy - q0["v"]) ** 2) < q0["r"] ** 2
                q = seed_in_mask(mask, n=36, seed=0)
                tr, vis = track_points(fr, q, device="cuda")
                cen = np.array([np.median(tr[t, vis[t]] if vis[t].mean() > 0.3 else tr[t],
                                         axis=0) for t in range(len(fr))])
                out[f"{role}_cen"] = cen; out[f"{role}_vis"] = vis
            np.savez(dst, **out)
            print(f"tracked {key}_seed{sd}: " +
                  " ".join(f"{r} {100*out[f'{r}_vis'].mean():.0f}%"
                           for r in ("subject", "partner") if f"{r}_vis" in out), flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "seeds":
        raise SystemExit(write_seeds())
    if cmd == "track":
        raise SystemExit(track())
    print(__doc__)
    raise SystemExit(2)
