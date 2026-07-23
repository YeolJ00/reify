"""Prep the realistic drop scene: simulate a real object dropping onto the table,
export its mesh (OBJ), per-frame poses, camera, and true params for the Blender
render + the recovery. (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.sim.diff_drop import DiffDrop  # noqa: E402

NAME = "Great_Dinos_Triceratops_Toy"
GROUND_Z = 0.706
TRUE = dict(density=800.0, cd=7.0, mu=0.5)    # cd = restitution damping
POS0 = [0.05, 0.0, 1.30]
VEL0 = [0.15, 0.0, 0.0]
ANG0 = [0.0, 0.0, 0.0]
DT, NSTEPS, STRIDE = 2.0e-4, 2400, 32        # 76 frames: fall + one bounce + rise
CAM = dict(eye=[0.90, -0.90, 1.12], target=[0.13, 0.0, 1.0], fov_deg=46, width=680, height=560)


def main():
    wp.init()
    out = REPO / "outputs" / "drop"
    out.mkdir(parents=True, exist_ok=True)
    with wp.ScopedDevice("cuda:0"):
        sim = DiffDrop(NAME, POS0, VEL0, ANG0, density=TRUE["density"], ground_z=GROUND_Z,
                       dt=DT, n_steps=NSTEPS, k=3000.0, cd=TRUE["cd"], mu=TRUE["mu"], requires_grad=False)
        sim.rollout()
        pos = sim.positions(STRIDE)
        ori = sim.orientations(STRIDE)
        V, F = sim.mesh_V, sim.mesh_F

    # export mesh OBJ (body frame, COM-centered) with smooth normals
    nrm = np.zeros_like(V)
    for tri in F:
        a, b, c = V[tri]
        n = np.cross(b - a, c - a); nrm[tri] += n
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    lines = ["# triceratops (body frame)"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in nrm]
    lines += [f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}" for a, b, c in F]
    (out / "object.obj").write_text("\n".join(lines))

    np.savez(out / "poses.npz", pos=pos, quat=ori)
    (out / "scene.json").write_text(json.dumps(dict(
        camera=CAM, ground_z=GROUND_Z, name=NAME, true=TRUE, stride=STRIDE, dt=DT,
        pos0=POS0, vel0=VEL0, ang0=ANG0, n_frames=len(pos)), indent=2))
    print(f"prepped {len(pos)} frames -> {out}")
    print(f"object rests: z {pos[0,2]:.3f} -> {pos[-1,2]:.3f}, slid {pos[-1,0]-pos[0,0]:+.3f} m")


if __name__ == "__main__":
    main()
