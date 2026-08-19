"""SPIN and STACK, rebuilt on the mesh-contact solver.

Both existed in the legacy `joint_fit.py` probe set but were last measured under the sphere
cover, the restart-based tilt ramp, and per-scene shared parameters, so none of their numbers
survived. Neither had ever been checked against a parameter sweep either -- which is the rule
that came out of `collide`, a probe built to recover mass whose observable is provably
mass-independent.

SPIN -- a body is spun about its vertical axis and left to grind to a halt. Friction at the
contact patch opposes the spin with a torque of roughly mu*m*g*r_eff, so the angular
deceleration goes as mu. The readout is the decay rate of |omega|, which is a magnitude, but a
magnitude taken WITHIN one object under a fixed excitation -- the regime where the bounce probe
measured rho = -0.995. The confound that killed magnitudes was comparing them across objects.

STACK -- one object on another, and the table is tilted until the stack comes apart. The top
body breaks away when tan(theta) exceeds the friction of the pair, so the observable is a
THRESHOLD angle rather than a distance, and it reads the object-on-object friction that no
other probe touches: every other contact in this project is against the table.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.sim.mesh_scene import MeshProbeScene                     # noqa: E402
from src.sim.tilt_probe import ramp_gravity_seq, onset_angle      # noqa: E402

GZ = 0.706
FPS, SUBSTEPS = 24.0, 60
TABLE_MU, TABLE_CD = 0.45, 3000.0

SPINNER = ("rigid/wooden_bowl_01", 0.62, 0.40)      # name, scale, mass kg
OMEGA0 = 14.0                                        # rad/s about z
# The top body must SLIDE, not tip, or the probe measures its geometry instead of the
# friction it is for: a bowl rests on a narrow foot ring and rocks off it at ~17 degrees
# whatever the friction, which pinned four of five sweep values to the same angle. The
# C-clamp has base/height 6.3 and lies flat. The base is scaled up so the top has room to
# slide without simply running off the edge -- at 1.0 the box top was 24x32 cm and a 19 cm
# bowl left it almost immediately.
BASE = ("rigid/cardboard_box_01", 1.10, 1.60)
TOP = ("rigid/Pony_C_Clamp_1440", 1.0, 1.10)
MUS = [0.15, 0.25, 0.40, 0.60, 0.85]


def eff_mu(mu, other=TABLE_MU):
    """The solver combines contact coefficients by geometric mean, so the angle a body
    actually slips at is atan(sqrt(mu_a*mu_b)), not atan(mu_a)."""
    return float(np.sqrt(mu * other))


def spin(mu, n_frames=48):
    name, sc, m = SPINNER
    s = MeshProbeScene([name], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
                       ang0=[[0.0, 0.0, OMEGA0]], masses=[m], ground_z=GZ,
                       dt=1.0 / (FPS * SUBSTEPS), n_steps=n_frames * SUBSTEPS,
                       k=2500.0, cd=[3000.0], mu=[mu], mesh_scale=[sc], faces=900,
                       ground_mu=TABLE_MU, ground_cd=TABLE_CD)
    p0 = s.pos0.copy(); p0[0, 2] = s.rest_height(0) + 0.002; s.pos0 = p0
    s.calibrate_stiffness()
    s.rollout()
    W = np.stack([s.vang[t].numpy()[0] for t in range(0, s.n_steps + 1, SUBSTEPS)])
    if not np.isfinite(W).all():
        return None
    wz = np.abs(W[:, 2])
    if wz[0] < 1e-6:
        return None
    # time to fall to 1/e of the initial spin, linearly interpolated between frames
    thr = wz[0] / np.e
    below = np.nonzero(wz <= thr)[0]
    if len(below) == 0:
        tau = float(n_frames / FPS)          # never got there within the window
        stopped = False
    else:
        i = int(below[0])
        if i == 0:
            tau, stopped = 0.0, True
        else:
            f = (wz[i - 1] - thr) / max(wz[i - 1] - wz[i], 1e-9)
            tau, stopped = float((i - 1 + f) / FPS), True
    return {"tau_s": tau, "w0": float(wz[0]), "w_end": float(wz[-1]), "stopped": stopped}


def stack(mu_pair, deg1=50.0, n_frames=64, settle=12):
    """Tilt until the top body breaks away from the one under it."""
    (bn, bs, bm), (tn, ts, tm) = BASE, TOP
    s = MeshProbeScene([bn, tn], [[0.0, 0.0, 0.0]] * 2, [[0.0, 0.0, 0.0]] * 2,
                       masses=[bm, tm], ground_z=GZ,
                       dt=1.0 / (FPS * SUBSTEPS), n_steps=n_frames * SUBSTEPS,
                       k=2500.0, cd=[3000.0, 3000.0], mu=[mu_pair, mu_pair],
                       mesh_scale=[bs, ts], faces=700,
                       ground_mu=0.9, ground_cd=TABLE_CD)   # base grips the table hard, so
    p0 = s.pos0.copy()                                       # the STACK is what gives way
    p0[0, 2] = s.rest_height(0) + 0.002
    base_top = p0[0, 2] + float(s._loc_np[s._bod_np == 0][:, 2].max())
    p0[1, 2] = base_top - float(s._loc_np[s._bod_np == 1][:, 2].min()) + 0.002
    s.pos0 = p0
    s.calibrate_stiffness()
    s.gravity_seq = ramp_gravity_seq(0.0, deg1, n_frames, SUBSTEPS, settle)
    s.rollout()
    P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
    if not np.isfinite(P).all():
        return None
    # Displacement of the top IN THE BASE'S OWN FRAME. Taking (top - base) in world
    # coordinates counts the base's rotation as sliding, and the base rotates with the ramp
    # by construction -- so every case registered a "collapse" as soon as the table moved.
    rel = np.zeros((len(P), 2))
    for t in range(len(P)):
        x, y, z, w = Q[t, 0]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        rel[t] = (R.T @ (P[t, 1] - P[t, 0]))[:2]
    ang = onset_angle(rel, 0.0, deg1, n_frames, settle, s.sizes[1], frac=0.05)
    slid = float(np.linalg.norm(rel[-1] - rel[settle]) * 100)
    creep = float(np.linalg.norm(rel[settle] - rel[0]) * 100)   # motion during settling
    return {"collapse_deg": ang, "rel_slide_cm": slid, "settle_creep_cm": creep}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or os.environ.get("LAB", "outputs/scene/spinstack"))
    out.mkdir(parents=True, exist_ok=True)
    wp.init()
    res = {"spin": [], "stack": []}
    with wp.ScopedDevice("cuda:0"):
        print(f"=== SPIN  ({SPINNER[0].split('/')[-1]} spun at {OMEGA0:.0f} rad/s about z)")
        print(f"  {'mu':>6}{'mu_eff':>8}{'tau to 1/e':>12}{'w_end':>9}")
        for mu in MUS:
            r = spin(mu)
            if r is None:
                print(f"  {mu:>6.2f}   DIVERGED"); continue
            r["mu"] = mu; r["mu_eff"] = eff_mu(mu); res["spin"].append(r)
            print(f"  {mu:>6.2f}{r['mu_eff']:>8.3f}{r['tau_s']:>11.3f}s{r['w_end']:>9.2f}")
        sp = res["spin"]
        if len(sp) > 2:
            c = np.array([[x["mu_eff"], x["tau_s"]] for x in sp])
            print(f"  rho(mu_eff, spin-down time) = {np.corrcoef(c[:,0], c[:,1])[0,1]:+.3f}"
                  f"   (expect NEGATIVE: more friction, stops sooner)")

        print(f"\n=== STACK ({TOP[0].split('/')[-1]} on {BASE[0].split('/')[-1]}, ramp to 50 deg)")
        print(f"  {'mu':>6}{'mu_pair':>9}{'predicted':>11}{'collapse':>10}{'err':>7}"
              f"{'slide cm':>10}{'creep':>9}")
        for mu in MUS:
            r = stack(mu)
            if r is None:
                print(f"  {mu:>6.2f}   DIVERGED"); continue
            pred = float(np.degrees(np.arctan(mu)))    # both bodies carry the same mu here
            r["mu"] = mu; r["pred_deg"] = pred; res["stack"].append(r)
            cd_ = r["collapse_deg"]
            print(f"  {mu:>6.2f}{mu:>9.2f}{pred:>11.1f}"
                  f"{(f'{cd_:.1f}' if cd_ else 'none'):>10}"
                  f"{(f'{cd_-pred:+.1f}' if cd_ else '--'):>7}{r['rel_slide_cm']:>10.1f}"
                  f"{r['settle_creep_cm']:>9.2f}")
        st = [x for x in res["stack"] if x["collapse_deg"]]
        if len(st) > 2:
            c = np.array([[x["pred_deg"], x["collapse_deg"]] for x in st])
            e = c[:, 1] - c[:, 0]
            print(f"  rho(predicted, measured) = {np.corrcoef(c[:,0], c[:,1])[0,1]:+.3f}"
                  f"   bias {e.mean():+.2f} deg   residual {np.abs(e-e.mean()).mean():.2f} deg")
    json.dump(res, open(out / "spinstack.json", "w"), indent=1)
    print(f"\n  wrote {out}/spinstack.json")


if __name__ == "__main__":
    main()
