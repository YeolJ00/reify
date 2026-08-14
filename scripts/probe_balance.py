"""PAN BALANCE -- mass ratio as a THRESHOLD.

Every previous attempt at mass failed because the observable did not depend on it. A body
shoved at fixed velocity slides v^2/(2*mu*g), which is mass-independent; that probe could
not have worked however it was tuned.

A balance can. A beam on a frictionless pivot tips toward whichever side carries the larger
moment, so the observable is the SIGN of

    m_obj * d_obj  -  m_ref * d_ref

which is a threshold, not a magnitude -- immune to the motion-magnitude confound that has
tracked every other reading in this project (rho up to -0.88). Sweeping the reference mass
brackets the unknown mass to whatever resolution is sampled.

The pivot is an exact reduced-coordinate revolute constraint (mesh_contact.apply_revolute),
not a penalty spring: a spring would add a compliance whose stiffness biases the tipping
threshold, which is the kind of knob that turned the tilt detector into a calibration step.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.sim.mesh_scene import MeshProbeScene          # noqa: E402

GZ = 0.706
FPS, SUBSTEPS, NF = 24.0, 60, 48
BEAM_L, BEAM_W, BEAM_T = 0.60, 0.05, 0.012      # m
PAN_R, PAN_T, PAN_RIM = 0.085, 0.022, 0.035
ARM = 0.24                                       # pivot -> pan centre
PIVOT_H = 0.14                                   # pivot height above the table


def beam_with_pans():
    """Beam and both pans as ONE rigid body.

    Modelling the pans as separate free bodies resting on the beam does not work: the moment
    the beam tilts they slide off it, and the tilt sign then reports where the pans went
    rather than which side is heavier (measured 2/7 correct). A real balance has its pans
    fixed to the beam, so the rig should too -- and as one body there is nothing to slide.

    Each pan is a shallow CUP (floor plus a rim), so the weight sitting in it cannot walk off
    the edge as the beam swings. Without the rim the probe measures friction between the
    weight and the pan, which is not the parameter it is for.
    """
    parts = [trimesh.creation.box(extents=(BEAM_L, BEAM_W, BEAM_T))]
    for sgn in (-1.0, +1.0):
        floor = trimesh.creation.cylinder(radius=PAN_R, height=PAN_T, sections=28)
        floor.apply_translation([sgn * ARM, 0.0, BEAM_T / 2 + PAN_T / 2])
        parts.append(floor)
        rim = trimesh.creation.annulus(r_min=PAN_R * 0.86, r_max=PAN_R,
                                       height=PAN_RIM, sections=28)
        rim.apply_translation([sgn * ARM, 0.0, BEAM_T / 2 + PAN_T + PAN_RIM / 2])
        parts.append(rim)
    return trimesh.util.concatenate(parts)


def run(m_obj, m_ref, obj_asset=None, obj_scale=1.0, n_frames=NF):
    """One weighing. Returns the beam's tilt angle over time (deg, + = object side down)."""
    z_beam = GZ + PIVOT_H
    # pan floor top, in world z. Weights are placed by their CENTRE, so half the weight's
    # own height must be added -- otherwise each weight starts buried inside the pan floor
    # and the contact spends the run pushing it out instead of weighing it.
    W = 0.05
    z_pan = z_beam + BEAM_T / 2 + PAN_T + W / 2 + 0.002
    names = [beam_with_pans()]
    masses = [0.45]
    pos = [[0.0, 0.0, z_beam]]
    # unknown on the +x pan, reference on the -x pan
    obj = obj_asset if obj_asset is not None else trimesh.creation.box(extents=(W,) * 3)
    names.append(obj); masses.append(m_obj); pos.append([+ARM, 0.0, z_pan])
    names.append(trimesh.creation.box(extents=(W,) * 3)); masses.append(m_ref)
    pos.append([-ARM, 0.0, z_pan])

    s = MeshProbeScene(names, pos, [[0., 0., 0.]] * len(names), masses=masses,
                       ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS), n_steps=n_frames * SUBSTEPS,
                       k=2500.0, cd=3000.0, mu=0.6, faces=600,
                       mesh_scale=[1.0, obj_scale, 1.0],
                       ground_mu=0.45, ground_cd=3000.0)
    s.calibrate_stiffness()
    # The beam must be STIFF. calibrate_stiffness sizes k from the body's own length, and a
    # 0.60 m beam gets a soft spring; in series with the weight (kij = ki*kj/(ki+kj)) that
    # gave ~18 mm of penetration against an 8 mm pan floor, so every weight sank straight
    # through its pan and landed on the table -- measured: weights settling at z=0.731, which
    # is table height plus half a box. The beam then weighs nothing and its angle is noise.
    kk = s.k.numpy(); kk[0] = 2.0e5; s.k.assign(kk)
    s.set_hinge(0, anchor=(0.0, 0.0, z_beam), axis=(0.0, 1.0, 0.0))
    s.rollout()
    Q = s.rotations(SUBSTEPS)[:, 0]          # beam quaternion per frame
    # signed rotation about +y, in degrees. +y rotation drops +x (the object side).
    ang = np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0)))
    return ang, s


def settled(ang, frac=0.25):
    """Equilibrium tilt: mean over the last `frac` of the run, not a single frame.

    Even with pivot damping the beam arrives with some residual swing; averaging the tail
    reads the equilibrium the moments actually define instead of one sample of the ringing.
    """
    n = max(int(len(ang) * frac), 1)
    return float(np.mean(ang[-n:]))


def main():
    out = Path(os.environ.get("LAB", "outputs/scene/balance")); out.mkdir(parents=True, exist_ok=True)
    wp.init()
    M_OBJ = 0.30
    REFS = [0.10, 0.20, 0.26, 0.30, 0.34, 0.40, 0.60]
    res = []
    print("=== DEPENDENCE CHECK: does the tilt sign track the moment difference?")
    print(f"  unknown mass fixed at {M_OBJ:.2f} kg, sweeping the reference")
    print(f"  {'m_ref':>7}{'expect':>9}{'final tilt':>12}{'sign':>7}{'ok':>5}")
    with wp.ScopedDevice("cuda:0"):
        for mr in REFS:
            ang, _ = run(M_OBJ, mr)
            fin = settled(ang)
            expect = "obj down" if M_OBJ > mr else ("ref down" if M_OBJ < mr else "balance")
            got = "obj down" if fin > 1.0 else ("ref down" if fin < -1.0 else "balance")
            ok = (expect == got) or (abs(M_OBJ - mr) < 0.02)
            res.append({"m_ref": mr, "m_obj": M_OBJ, "tilt_deg": fin,
                        "expect": expect, "got": got, "ok": bool(ok),
                        "ang": [float(a) for a in ang]})
            print(f"  {mr:>7.2f}{expect:>9}{fin:>12.2f}{got:>7}{str(ok):>5}"
                  .replace("obj down", "obj").replace("ref down", "ref"))
    n_ok = sum(r["ok"] for r in res)
    print(f"  {n_ok}/{len(res)} correct")
    # bracket the unknown mass from the sign flip
    lo = max([r["m_ref"] for r in res if r["tilt_deg"] > 1.0], default=None)
    hi = min([r["m_ref"] for r in res if r["tilt_deg"] < -1.0], default=None)
    print(f"\n=== RECOVERY: sign flip brackets the unknown mass")
    print(f"  true {M_OBJ:.3f} kg   bracketed [{lo}, {hi}]")
    json.dump({"m_obj": M_OBJ, "sweep": res, "bracket": [lo, hi]},
              open(out / "balance.json", "w"), indent=1)
    print(f"  wrote {out}/balance.json")


if __name__ == "__main__":
    main()
