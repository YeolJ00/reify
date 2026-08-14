"""Two damping probes, both sweeping cd to prove dependence before reporting recovery.

BOUNCE -- magnitude form, and that is the finding. Rebound HEIGHT already failed (rho = -0.186): it is a magnitude,
and magnitudes here are dominated by shape, so a bowl and a book differ regardless of their
damping. The threshold version -- drop it near the EDGE and ask whether it leaves the table -- does not
work: at a 0.22 m/s nudge nothing leaves (0/6), at 0.52 everything does (6/6). The transition
is governed by the nudge, not by damping, so there is no cd threshold to find.

The magnitude version does work, and cleanly: rho(log cd, rebound) = -0.995 over a 32x sweep.
The earlier -0.186 was measured ACROSS DIFFERENT OBJECTS, where shape dominates. Holding the
object and the excitation fixed and sweeping only cd, rebound height is an excellent reading.
The confound was never "magnitudes are bad" -- it was comparing magnitudes across objects.

SETTLE -- decay form. Tip the bowl onto its dome and release, then read the decay
rate of the rocking: rho(log cd, decay) = +0.974.

The frequency half does NOT work. The dominant FFT bin sits at the lowest non-zero frequency
for every candidate, i.e. the window length, which means the motion is not oscillating enough
for a period to exist -- so the amplitude-independence argument, which was the reason to want a
frequency observable, cannot be tested here and is not claimed.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.sim.mesh_scene import MeshProbeScene          # noqa: E402

GZ = 0.706
FPS, SUBSTEPS = 24.0, 60
TABLE = (0.79, 0.49)                       # half-extents, matches the rendered table
CDS = [400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0]
OBJ = "rigid/wooden_bowl_01"
SCALE, MASS = 0.62, 0.40


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def quat_y(deg):
    h = np.radians(deg) / 2.0
    return [0.0, float(np.sin(h)), 0.0, float(np.cos(h))]


def build(cd, n_frames, pos, vel=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0), rot=None):
    s = MeshProbeScene([OBJ], [list(pos)], [list(vel)], ang0=[list(ang)],
                       rot0=None if rot is None else [rot], masses=[MASS],
                       ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS), n_steps=n_frames * SUBSTEPS,
                       k=2500.0, cd=[cd], mu=[0.35], mesh_scale=[SCALE], faces=900,
                       ground_mu=0.45, ground_cd=3000.0, table=TABLE)
    s.calibrate_stiffness()
    return s


def bounce(cd, n_frames=48):
    """Dropped near the edge with a nudge toward it. On the table, or off?"""
    # A gentle nudge only. At 0.9 m/s every candidate slid off whatever its damping (6/6),
    # so the probe measured the shove, not restitution. The horizontal push must be small
    # enough that whether the object clears the edge is decided by how much it BOUNCES.
    s = build(cd, n_frames, pos=(TABLE[0] - 0.11, 0.0, GZ + 0.22), vel=(0.22, 0.0, 0.0))
    s.rollout()
    P = s.positions(SUBSTEPS)[:, 0]
    if not np.isfinite(P).all():
        return None
    off = bool(P[-1, 2] < GZ - 0.05)                     # fell past the table top
    z = P[:, 2]
    lo = int(np.argmin(z))
    reb = float(z[lo:].max() - z[lo])
    return {"off_table": off, "rebound_m": reb, "x_final": float(P[-1, 0])}


def settle(cd, n_frames=120):
    """Rocked about y; read the dominant frequency and the decay of the angular signal."""
    # Controlled DISPLACEMENT, not a velocity kick: tip the bowl 28 deg and release. Every
    # candidate then starts at the same amplitude, which is what makes the amplitude-vs-period
    # independence testable rather than assumed.
    # UPSIDE DOWN, on its dome. Tipped on its flat foot the bowl simply falls back and
    # stops -- no oscillation at all, so the FFT returned the lowest bin for every candidate
    # and the "frequency" was an artifact of the window length. Inverted, the curved outside
    # is the contact surface and the bowl rocks like a rocking chair, which is the only
    # configuration in which a period exists to be measured.
    s = build(cd, n_frames, pos=(0.0, 0.0, GZ + 0.075),
              rot=quat_mul(quat_y(180.0), quat_y(22.0)))
    s.rollout()
    Q = s.rotations(SUBSTEPS)[:, 0]
    if not np.isfinite(Q).all():
        return None
    a = np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0)))
    a = a - a.mean()
    w = np.hanning(len(a))
    sp = np.abs(np.fft.rfft(a * w))
    fr = np.fft.rfftfreq(len(a), d=1.0 / FPS)
    sp[0] = 0.0
    f_dom = float(fr[int(np.argmax(sp))])
    env = np.abs(a)
    half = len(env) // 2
    e0, e1 = env[:half].max(), env[half:].max()
    decay = float(np.log((e0 + 1e-9) / (e1 + 1e-9)) / (half / FPS))   # 1/s
    return {"f_hz": f_dom, "decay_per_s": decay, "amp0": float(e0)}


def main():
    out = Path(os.environ.get("LAB", "outputs/scene/damping")); out.mkdir(parents=True, exist_ok=True)
    wp.init()
    res = {"bounce": [], "settle": []}
    with wp.ScopedDevice("cuda:0"):
        print("=== BOUNCE, dropped 20 cm near the table edge and nudged toward it")
        print(f"  {'cd':>8}{'off table':>11}{'rebound cm':>12}{'x final':>9}")
        for cd in CDS:
            r = bounce(cd)
            if r is None:
                print(f"  {cd:>8.0f}   DIVERGED"); continue
            r["cd"] = cd; res["bounce"].append(r)
            print(f"  {cd:>8.0f}{str(r['off_table']):>11}{r['rebound_m']*100:>12.2f}"
                  f"{r['x_final']:>9.3f}")
        b = res["bounce"]
        if b:
            offs = [x["off_table"] for x in b]
            thr = [b[i]["cd"] for i in range(1, len(b)) if offs[i] != offs[i - 1]]
            c = np.array([[x["cd"], x["rebound_m"]] for x in b])
            print(f"  threshold form: leaves the table for cd below {thr[0] if thr else 'n/a'}"
                  f"  ({sum(offs)}/{len(offs)} off)")
            print(f"  magnitude form: rho(log cd, rebound) = "
                  f"{np.corrcoef(np.log10(c[:,0]), c[:,1])[0,1]:+.3f}")

        print("\n=== SETTLE, rocked 3.5 rad/s about y")
        print(f"  {'cd':>8}{'f dom Hz':>11}{'decay 1/s':>11}{'amp0 deg':>10}")
        for cd in CDS:
            r = settle(cd)
            if r is None:
                print(f"  {cd:>8.0f}   DIVERGED"); continue
            r["cd"] = cd; res["settle"].append(r)
            print(f"  {cd:>8.0f}{r['f_hz']:>11.2f}{r['decay_per_s']:>11.2f}{r['amp0']:>10.2f}")
        st = res["settle"]
        if st:
            c = np.array([[x["cd"], x["f_hz"], x["decay_per_s"], x["amp0"]] for x in st])
            print(f"  rho(log cd, decay)     = {np.corrcoef(np.log10(c[:,0]), c[:,2])[0,1]:+.3f}"
                  f"   <- the reading")
            print(f"  rho(log cd, frequency) = {np.corrcoef(np.log10(c[:,0]), c[:,1])[0,1]:+.3f}")
            print(f"  rho(amp0, frequency)   = {np.corrcoef(c[:,3], c[:,1])[0,1]:+.3f}"
                  f"   <- must be ~0: period is amplitude-free")
    json.dump(res, open(out / "damping.json", "w"), indent=1)
    print(f"\n  wrote {out}/damping.json")


if __name__ == "__main__":
    main()
