"""Validate downloaded assets: load every mesh, print stats, then prove
sim-ability by dropping one rigid asset onto a ground plane in Newton.

Usage: python scripts/inspect_assets.py [--drop Schleich_Lion_Action_Figure]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.assets import add_rigid_asset, asset_stats, list_assets, load_asset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="Schleich_Lion_Action_Figure",
                    help="rigid asset to drop-test in Newton ('' to skip)")
    args = ap.parse_args()

    inv = list_assets()
    print("asset inventory:", {k: len(v) for k, v in inv.items()})
    for cat in ("rigid", "cloth", "scenes"):
        for name in inv[cat]:
            try:
                m = load_asset(cat, name)
                s = asset_stats(m)
                print(f"  [{cat}] {name}: {s['vertices']}v/{s['faces']}f "
                      f"extent={s['extent_m']} watertight={s['watertight']}")
            except Exception as e:
                print(f"  [{cat}] {name}: LOAD FAILED — {e}")

    if not args.drop:
        return 0

    print(f"\ndrop test: {args.drop}")
    import warp as wp

    import newton

    wp.init()
    with wp.ScopedDevice("cuda:0" if wp.get_device("cuda:0").is_cuda else "cpu"):
        m = load_asset("rigid", args.drop)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        add_rigid_asset(builder, m, pos=(0.0, 0.0, 0.5))
        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model)
        state_a, state_b = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        dt = 1.0 / 240.0
        for step in range(480):  # 2 s
            state_a.clear_forces()
            pipeline.collide(state_a, contacts)
            solver.step(state_a, state_b, None, contacts, dt)
            state_a, state_b = state_b, state_a

        q = state_a.body_q.numpy()[0]
        print(f"final body pose: pos={np.round(q[:3], 4)} (z should settle near object half-height)")
        assert np.isfinite(q).all(), "drop test produced non-finite pose"
        assert q[2] > -0.05, "object fell through the ground"
        print("drop test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
