"""J0: render three clips from clearly different theta and check they differ.

Stiffness is the knob that should read as "material character" to a judge: a very stiff
cloth behaves like a board, a very floppy one ripples and folds. If these three do not look
different to a human, no judge will separate them and J2 cannot pass.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/j0_render_check.py    (warp env)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.render.clip import probe, render_frames, theta_key, write_mp4  # noqa: E402
from src.sim.rollout import FlagSim  # noqa: E402

OUT = REPO / "outputs" / "judge" / "j0"
SECONDS = 3.0
# stiffness spans ~2.5 decades; damping raised with it so the stiff case stays stable
THETAS = {
    "stiff":  {"tri_ke": 5.0e4, "tri_kd": 40.0, "mass": 0.05},
    "medium": {"tri_ke": 5.0e3, "tri_kd": 10.0, "mass": 0.05},
    "floppy": {"tri_ke": 2.0e2, "tri_kd": 2.0,  "mass": 0.05},
}


def substeps_for(ke, mass, fps, kd=10.0, cfl=0.15, damp_cfl=0.09, floor=32):
    """Substeps the explicit solver needs to stay stable at this stiffness AND damping.

    TWO conditions, not one. The config warns about both -- dt*sqrt(ke/m) for the stretch
    force and kd/m*dt for the damping -- and implementing only the first sent me chasing
    the wrong parameter: raising substeps for stiffness did not stop the "stiff" flag
    exploding, because it was the kd=40 damping doing it. Measured directly at 36
    substeps, kd=10 is stable and kd=20 is not, while stiffness runs to ke=4e4 happily
    once its own condition is met.

    The floor is the config's own 32 and it is a floor, not a starting point: dropping to
    the 22 the stretch condition alone allowed blew up a flag that is stable at 32.
    """
    import math
    need_k = math.sqrt(float(ke) / float(mass)) / (cfl * float(fps))
    need_d = float(kd) / (damp_cfl * float(mass) * float(fps))
    return max(int(math.ceil(need_k)), int(math.ceil(need_d)), int(floor))


def build(cfg, theta, seconds):
    c = dict(cfg)
    c["cloth"] = dict(cfg["cloth"]); c["sim"] = dict(cfg["sim"])
    c["cloth"]["tri_ke"] = theta["tri_ke"]; c["cloth"]["tri_ka"] = theta["tri_ke"]
    c["cloth"]["tri_kd"] = theta["tri_kd"]; c["cloth"]["mass"] = theta["mass"]
    c["sim"]["num_frames"] = int(seconds * cfg["sim"]["fps"])
    c["sim"]["substeps"] = substeps_for(theta["tri_ke"], theta["mass"], cfg["sim"]["fps"],
                                        kd=theta["tri_kd"])
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / "configs" / "flag.yaml"))
    cam = cfg["m4"]["camera"]
    eye, tgt = cam["eye"], cam["target"]
    rows = {}
    for name, th in THETAS.items():
        c = build(cfg, th, SECONDS)
        sub = c["sim"]["substeps"]
        t0 = time.time()
        sim = FlagSim(c); sim.rollout()
        X = sim.trajectory()
        t1 = time.time()
        # render every 2nd frame: 60 fps sim -> 30 fps clip, 3 s
        frames = render_frames(sim, eye, tgt, width=640, height=480, every=2)
        t2 = time.time()
        key = theta_key(th, "flag", cfg["seed"])
        path = OUT / f"{name}_{key}.mp4"
        write_mp4(frames, path, fps=30)
        # how much does the cloth actually move? free corner path length
        tip = X[:, -1, :]
        travel = float(np.linalg.norm(np.diff(tip, axis=0), axis=1).sum())
        span = float(np.linalg.norm(tip.max(0) - tip.min(0)))
        rows[name] = {"theta": th, "clip": path.name, "sim_s": round(t1 - t0, 2),
                      "render_s": round(t2 - t1, 2), "frames": int(len(frames)),
                      "tip_path_m": round(travel, 3), "tip_span_m": round(span, 3),
                      "probe": probe(path)}
        stable = span < 5.0
        rows[name]["substeps"] = sub
        rows[name]["stable"] = bool(stable)
        print(f"  {name:7s} ke={th['tri_ke']:>8.0f} substeps={sub:3d}  sim {t1-t0:5.2f}s  "
              f"render {t2-t1:5.2f}s  tip path {travel:6.3f} m  span {span:5.3f} m  "
              f"{'' if stable else 'UNSTABLE'}  -> {path.name}")

    # do the clips differ as pixels? (a judge cannot separate what a difference cannot)
    print("\n  pairwise mean abs frame difference between clips:")
    import itertools
    keys = list(rows)
    vids = {}
    for k in keys:
        import imageio.v2 as imageio
        vids[k] = np.stack([f for f in imageio.mimread(OUT / rows[k]["clip"],
                                                       memtest=False)])[..., :3]
    for a, b in itertools.combinations(keys, 2):
        n = min(len(vids[a]), len(vids[b]))
        d = np.abs(vids[a][:n].astype(int) - vids[b][:n].astype(int)).mean()
        print(f"    {a:7s} vs {b:7s}: {d:6.2f} / 255")
    (OUT / "j0_report.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}/j0_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
