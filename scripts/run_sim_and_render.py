"""One command: simulate in Warp/Newton, render in Blender, build the comparison.

Two processes, not one, and for a concrete reason: Blender bundles its own Python and
cannot import the warp environment. They exchange exactly one artefact -- sim_poses.json,
a per-frame object transform -- which is the whole interface between the physics and the
rendering layers.

    warp env     rollout at the recovered parameter  -> sim_poses.json
    blender      drives the staged scene with it     -> sim_<key>/f####.png
    warp env     pairs them with the generated clip  -> docs/cmp_<key>.webp

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/run_sim_and_render.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/home/jooyeolyun/anaconda3/envs/warp/bin/python"
BLENDER = "/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender"
LAB = REPO / "outputs" / "scene" / "expand"

STEPS = [
    ("simulate  (Warp)", [PY, "scripts/export_sim_poses.py"]),
    ("render    (Blender/Cycles)",
     [BLENDER, "--background", "--python", "scripts/blender_render_sim.py"]),
    ("compare   (pair with the generated clips)",
     [PY, "scripts/make_comparison_figs.py"]),
]


def main():
    env = dict(os.environ)
    for name, cmd in STEPS:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        tail = [ln for ln in (r.stdout or "").splitlines()
                if ln.strip() and not ln.startswith(("Module", "Warp", "   ", "Fra:"))]
        for ln in tail[-6:]:
            print("   ", ln)
        if r.returncode != 0:
            print(f"    FAILED ({r.returncode})")
            print((r.stderr or "")[-1200:])
            return r.returncode
        print(f"    ok, {time.time()-t0:.0f}s")
    print(f"\nwrote {len(list((REPO/'docs').glob('cmp_*.webp')))} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
