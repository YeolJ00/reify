"""Export a waving-flag mesh (one frame of the cloth sim) to OBJ, for placing in
a Blender city scene. Picks a mid-rollout frame so the flag is billowing.

Usage: python scripts/export_flag_obj.py [--frame 45] [--out outputs/flag_frame.obj]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.rollout import FlagSim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=45)
    ap.add_argument("--out", default=str(REPO / "outputs" / "flag_frame.obj"))
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "flag.yaml").read_text())
    wp.init()
    with wp.ScopedDevice(cfg["device"]):
        sim = FlagSim(cfg, requires_grad=False)
        sim.rollout()
        states = sim.frame_states()
        f = min(args.frame, len(states) - 1)
        verts = states[f].particle_q.numpy()
        faces = sim.model.tri_indices.numpy()

    # normals for smooth shading
    normals = np.zeros_like(verts)
    for tri in faces:
        a, b, c = verts[tri]
        n = np.cross(b - a, c - a)
        normals[tri] += n
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    lines = ["# flag mesh exported from cloth sim (waving frame)"]
    for v in verts:
        lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
    for n in normals:
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
    for tri in faces:
        a, b, c = tri + 1
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    Path(args.out).write_text("\n".join(lines))
    print(f"exported {len(verts)} verts, {len(faces)} faces -> {args.out}")
    print(f"flag bounds: {np.round(verts.min(0), 3)} .. {np.round(verts.max(0), 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
