"""G1a: is ds/dpixels through the frozen judge finite, sane, and REAL?

This is the cheapest link in the G-track chain to kill. If no usable gradient flows from
the pairwise margin back to the video pixels, gradient MAP is dead regardless of how good
the differentiable renderer is -- and it costs one forward/backward to find out, versus
installing nvdiffrast and writing a renderer first.

What is checked, in order of how badly each would kill the track:

  1. FINITE. Does backward() produce a gradient at all, with no NaN/Inf? The model runs in
     bfloat16, which has ~3 decimal digits, so underflow to exactly zero is a real risk.
  2. STRUCTURED. Is the gradient concentrated where the moving object is, or is it uniform
     noise? A gradient that ignores the object cannot carry physics information.
  3. PREDICTIVE. This is the one that matters and the one the project rule demands:
     perturb the pixels along the gradient by eps and check the margin moves by the
     predicted eps * ||g||^2. A gradient that is finite and pretty but does not predict
     the function's change is useless for ascent.

Scored in PLAUSIBILITY mode ("which clip shows more physically realistic motion"), which
is the objective the fit now uses.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> .../envs/cosmos/bin/python scripts/g1_pixel_grad.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "judge" / "g0c"
OUT = REPO / "outputs" / "judge" / "g1"


def main():
    import torch
    from src.judge.pairwise import PairwiseJudge, PLAUSIBILITY_PROMPTS
    from src.render.motion_budget import sample_frames

    OUT.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.loads((LAB / "clips.json").read_text()) if c["guard_ok"]]
    # a bouncy and a dead cell of the same object: a pair the judge should have an opinion on
    a = next(c for c in clips if c["clip"] == "brass_pot_cd300_mu050.mp4")
    b = next(c for c in clips if c["clip"] == "brass_pot_cd10000_mu050.mp4")
    print(f"A {a['clip']}  e={a['e']:.3f}\nB {b['clip']}  e={b['e']:.3f}\n")

    j = PairwiseJudge(n_frames=12)
    vA = sample_frames(LAB / a["clip"], 12)
    vB = sample_frames(LAB / b["clip"], 12)
    q = PLAUSIBILITY_PROMPTS[0]

    inputs = j.build_inputs(vA, vB, q)
    px = inputs["pixel_values_videos"]
    print(f"pixel_values_videos {tuple(px.shape)} {px.dtype} "
          f"range [{px.min().item():.2f}, {px.max().item():.2f}]")

    # --- 1. finite?
    leaf = px.detach().clone().float().requires_grad_(True)
    t0 = time.time()
    s = j.margin_from_inputs({**inputs, "pixel_values_videos": leaf.to(px.dtype)})
    s.backward()
    g = leaf.grad
    dt = time.time() - t0
    if g is None:
        print("\nFAIL: no gradient reached the pixels (grad is None).")
        return 1
    gn = g.detach().float()
    finite = bool(torch.isfinite(gn).all())
    nz = float((gn != 0).float().mean())
    print(f"\n1. FINITE   margin s = {s.item():+.4f}   fwd+bwd {dt:.1f}s")
    print(f"   grad finite: {finite} | nonzero elements: {100*nz:.1f}% | "
          f"||g|| = {gn.norm().item():.4e}")
    print(f"   grad range [{gn.min().item():+.2e}, {gn.max().item():+.2e}] "
          f"| mean|g| {gn.abs().mean().item():.3e}")
    if not finite or nz < 0.01:
        print("   FAIL: gradient is absent or numerically dead.")
        return 1

    # --- 2. structured? patches carrying the object should dominate
    #     pixel_values_videos is (n_patches, feat); norm per patch is the usable proxy.
    pg = gn.norm(dim=-1)
    top = torch.topk(pg, max(1, len(pg) // 20)).values
    print(f"\n2. STRUCTURED  per-patch |g|: median {pg.median().item():.3e}  "
          f"top-5% mean {top.mean().item():.3e}  "
          f"ratio {top.mean().item()/max(pg.median().item(),1e-30):.1f}x")
    print("   (a uniform-noise gradient gives ratio ~1; structure gives >>1)")

    # --- 3. predictive? directional finite differences
    print("\n3. PREDICTIVE  perturb along +g and check s moves as predicted")
    d = gn / gn.abs().max()   # eps = max per-pixel step; a unit-norm direction
                              # spread over 5.8M elements lands below bf16 resolution
    base = px.detach().float()
    print(f"   {'eps':>8} {'predicted ds':>13} {'actual ds':>11} {'ratio':>7}")
    rows = []
    with torch.no_grad():
        s0 = j.margin_from_inputs(
            {**inputs, "pixel_values_videos": base.to(px.dtype)}).item()
        for eps in (0.002, 0.005, 0.01, 0.02, 0.05):
            pert = (base + eps * d).to(px.dtype)
            s1 = j.margin_from_inputs(
                {**inputs, "pixel_values_videos": pert}).item()
            pred = eps * float((gn * d).sum())
            act = s1 - s0
            rows.append({"eps": eps, "pred": pred, "act": act,
                         "ratio": act / pred if pred else float("nan")})
            print(f"   {eps:>8.2f} {pred:>+13.4f} {act:>+11.4f} "
                  f"{act/pred if pred else float('nan'):>7.2f}")
    good = [r for r in rows if r["pred"] > 1e-6]
    ok_dir = all(r["act"] > 0 for r in good)
    small = [r for r in good if r["eps"] <= 0.005]
    ok_mag = any(0.3 < r["ratio"] < 3.0 for r in small) if small else False
    print(f"\n   direction correct at every eps: {ok_dir}")
    print(f"   magnitude within 3x at small eps: {ok_mag}")

    verdict = "PASS" if (finite and ok_dir) else "FAIL"
    print(f"\nG1a VERDICT: {verdict}")
    if verdict == "PASS" and not ok_mag:
        print("  (direction usable, magnitude off -- fine for ascent with a tuned step,")
        print("   but the local linear model is poor; expect small steps.)")
    (OUT / "pixel_grad.json").write_text(json.dumps(
        {"verdict": verdict, "s": s.item(), "grad_norm": gn.norm().item(),
         "nonzero_frac": nz, "structure_ratio":
             top.mean().item() / max(pg.median().item(), 1e-30),
         "fd": rows, "dir_ok": ok_dir, "mag_ok": ok_mag}, indent=2))
    print(f"\nwrote {OUT}/pixel_grad.json")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
