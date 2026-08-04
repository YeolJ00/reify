"""J1: score the J0 clips against two materials; report the matrix, throughput and VRAM.

This is a smoke signal, not the kill test. The question is only whether the ordering is
already sensible: a stiff flag should look more like a tarp than like silk, and a floppy
one the other way round.

Run: HF_HOME=... CUDA_VISIBLE_DEVICES=<g> \
     /home/jooyeolyun/anaconda3/envs/cosmos/bin/python scripts/j1_judge_throughput.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

J0 = REPO / "outputs" / "judge" / "j0"
OUT = REPO / "outputs" / "judge" / "j1"
MATERIALS = ["heavy stiff tarp", "light silk"]
ORDER = ["stiff", "medium", "floppy"]


def main():
    import torch
    from src.judge.cosmos import CosmosJudge

    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((J0 / "j0_report.json").read_text())
    clips = {k: J0 / rep[k]["clip"] for k in ORDER if k in rep}

    t0 = time.time()
    judge = CosmosJudge()
    load_s = time.time() - t0
    free, total = torch.cuda.mem_get_info()
    print(f"judge loaded in {load_s:.0f}s   VRAM used {torch.cuda.memory_allocated()/1e9:.1f} GB"
          f"   free {free/1e9:.1f}/{total/1e9:.1f} GB\n")

    res, n_calls = {}, 0
    t0 = time.time()
    for m in MATERIALS:
        res[m] = {}
        for k in ORDER:
            r = judge.score(str(clips[k]), m)
            res[m][k] = r
            n_calls += len(r["per_prompt"])
    dt = time.time() - t0
    per_clip_material = dt / (len(MATERIALS) * len(ORDER))

    print(f"{'material':20s}" + "".join(f"{k:>12s}" for k in ORDER))
    print("-" * (20 + 12 * len(ORDER)))
    for m in MATERIALS:
        print(f"{m:20s}" + "".join(f"{res[m][k]['score']:+12.3f}" for k in ORDER))
    print("-" * (20 + 12 * len(ORDER)))
    print(f"{'prompt spread':20s}" + "".join(f"{res[MATERIALS[0]][k]['spread']:12.3f}"
                                             for k in ORDER))

    # smoke signal: does tarp prefer stiff over floppy, and silk the reverse?
    tarp, silk = res["heavy stiff tarp"], res["light silk"]
    d_tarp = tarp["stiff"]["score"] - tarp["floppy"]["score"]
    d_silk = silk["floppy"]["score"] - silk["stiff"]["score"]
    print(f"\n  tarp prefers stiff over floppy by {d_tarp:+.3f}   "
          f"{'correct' if d_tarp > 0 else 'WRONG DIRECTION'}")
    print(f"  silk prefers floppy over stiff by {d_silk:+.3f}   "
          f"{'correct' if d_silk > 0 else 'WRONG DIRECTION'}")

    thr = 60.0 / per_clip_material
    print(f"\nthroughput: {per_clip_material:.2f} s per (clip, material) with "
          f"{len(res[MATERIALS[0]][ORDER[0]]['per_prompt'])} paraphrases "
          f"-> {thr:.1f} scored pairs/min  ({n_calls/dt:.1f} forward passes/s)")
    (OUT / "j1_matrix.json").write_text(json.dumps(
        {"scores": res, "seconds_per_pair": per_clip_material,
         "pairs_per_min": thr, "load_s": load_s}, indent=2))
    print(f"wrote {OUT}/j1_matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
