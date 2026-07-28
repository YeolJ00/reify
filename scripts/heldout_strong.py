"""Held-out prediction on a generated video that actually tests the material.

The first held-out attempt was inconclusive for a diagnosable reason: that clip's knock
was a glancing 18 px tap, and the fitted trajectory never even brought the balls into
contact, so the two candidate materials predicted the same thing to within 2.9 px. A
held-out test only tests what the held-out motion depends on — the same rule that
governs the fitting side.

So we hold out a video with a STRONG, causal collision instead (the target ball is
knocked 255 px, and the balls are genuinely touching when it starts moving). Material
recovered jointly from {drop, collide-seed2} is frozen; only the unknown launch
velocity of the held-out clip is fitted, because an initial condition cannot transfer
but a material must.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/heldout_strong.py    (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.render.camera import Camera  # noqa: E402
from scripts.cosmos_joint_heldout import DEFAULT, OUT, fit_v0_only, load, rms, simulate  # noqa: E402

HELD = ("collide", 4)          # strong causal knock, never used in the joint fit


def main():
    cfg = json.loads((OUT / "scene.json").read_text())
    cam = Camera(cfg["camera"])
    prev = np.load(OUT / "joint_heldout.npz", allow_pickle=True)
    th = json.loads(str(prev["theta"]))
    print(f"material recovered jointly from drop+collide(seed2): "
          f"cd={th['cd']:.2f} mu={th['mu']:.3f} ratio={th['ratio']:.3f}")

    wp.init()
    rng = np.random.default_rng(1)
    with wp.ScopedDevice("cuda:0"):
        probe, seed = HELD
        fr, A, B = load(probe, seed); nf = len(fr)
        mB = ~np.isnan(B[:, 0])
        print(f"held-out '{probe}' seed{seed}: target ball moves "
              f"{np.linalg.norm(B[mB][-1] - B[mB][0]):.0f} px in the video")

        r_rec, v_rec = fit_v0_only(cfg, cam, probe, th, A, B, nf, rng, iters=10, pop=20)
        r_def, v_def = fit_v0_only(cfg, cam, probe, dict(DEFAULT), A, B, nf, rng, iters=10, pop=20)
        Pr = simulate(cfg, probe, v_rec, th, nf)
        Pd = simulate(cfg, probe, v_def, dict(DEFAULT), nf)

        # does the SIM actually collide? (if the target never moves, the test is vacuous)
        movedB = np.linalg.norm(Pr[:nf, 1] - Pr[0, 1], axis=1).max()
        print(f"  sim target ball moves {movedB*100:.1f} cm  -> "
              f"{'contact happens, test is live' if movedB > 0.02 else 'NO CONTACT — test vacuous'}")
        print(f"  recovered material : {r_rec:5.1f} px   launch {np.round(v_rec,2)}")
        print(f"  default   material : {r_def:5.1f} px   launch {np.round(v_def,2)}   "
              f"-> {r_def / max(r_rec, 1e-9):.2f}x worse")

        # how far apart are the two predictions at all?
        uvrA, _ = cam.project(Pr[:nf, 0]); uvdA, _ = cam.project(Pd[:nf, 0])
        uvrB, _ = cam.project(Pr[:nf, 1]); uvdB, _ = cam.project(Pd[:nf, 1])
        sep = max(np.abs(uvrA - uvdA).max(), np.abs(uvrB - uvdB).max())
        print(f"  the two predictions differ by {sep:.1f} px "
              f"({'discriminating' if sep > 5 else 'too similar to discriminate'})")

    np.savez(OUT / "heldout_strong.npz", theta=json.dumps(th), r_rec=r_rec, r_def=r_def,
             Pr=Pr, Pd=Pd, A=A, B=B, probe=probe, seed=seed, sep=sep,
             default=json.dumps(DEFAULT))
    print(f"wrote {OUT}/heldout_strong.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
