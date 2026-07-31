"""Render each simulated rollout as frames of the actual mesh. No masks anywhere.

Reads sim_poses.json (per-frame pose from the rollout at the recovered parameter) and
rasterises the real mesh through the lab camera. Replaces the sprite compositing, whose
every defect came from a mask failing to match the object's shape.

Run: python scripts/render_sim_raster.py            (warp env, no GPU needed)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.assets import decimate, load_asset  # noqa: E402
from src.render.mesh_raster import render  # noqa: E402

LAB = REPO / "outputs" / "scene" / "expand"


def main():
    poses = json.loads((LAB / "sim_poses.json").read_text())
    cfg = json.loads((LAB / "lab.json").read_text())
    cam = dict(cfg["camera"]); GZ = cfg["ground_z"]
    out = {}
    for key, info in poses.items():
        subj = info["subject"]
        so = cfg["assets"][subj]
        cat, asset = so["asset"].split("/")[0], so["asset"].split("/")[-1]
        tm = decimate(load_asset(cat, asset), 1200).copy()
        tm.apply_scale(so["scale"])
        V = np.asarray(tm.vertices, float)
        V = V - V.mean(0)              # body frame: centred on the centroid, as the sim is
        F = np.asarray(tm.faces, int)
        frames = []
        for p in info["poses"]:
            R = np.asarray(p["mat"], float)
            # sim_poses stores the BLENDER origin; the mesh centroid is what we drew
            t = np.asarray(p["loc"], float) + R @ (np.asarray(tm.vertices).mean(0))
            frames.append(render(V, F, R, t, cam, ground_z=GZ))
        arr = np.stack(frames)
        np.savez_compressed(LAB / f"simr_{key}.npz", frames=arr, note=info["note"])
        out[key] = {"note": info["note"], "n": int(len(arr))}
        print(f"  {key}: {len(arr)} frames rendered  ({info['note']})")
    (LAB / "sim_render.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} rendered rollouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
