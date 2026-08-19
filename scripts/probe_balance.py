"""PAN BALANCE -- the mass of a REAL asset, read as a threshold.

Mass is not observable from sliding: a body shoved at fixed velocity travels v^2/(2*mu*g),
independent of its mass, which is why the earlier `collide` probe measured nothing. A balance
tips toward the larger moment, so the observable is the SIGN of

    m_obj * d_obj  -  m_ref * d_ref

a threshold rather than a magnitude, and immune to the motion-magnitude confound that tracks
every other reading in this project. Sweeping the reference brackets the unknown.

Mechanism
---------
Beam on an exact revolute joint (`mesh_contact.apply_revolute`) -- reduced-coordinate, no
stiffness parameter, so there is no compliance to bias the tipping threshold. Pans are part of
the beam, and the swing is bounded by stops at +/-9 degrees so the pans stay near level.

Hanging pans were tried and abandoned, which is worth recording because they are the more
realistic mechanism. `MeshProbeScene.add_link` (kept, and useful for any suspended assembly)
suspends each pan from the beam by a point-to-point spring attached above the pan's centre of
mass, so gravity levels it. It works as a joint, but a suspended pan is a pendulum, and its
swing threw the payload out: at a pan radius comfortable for an apple both payloads ended on
the table, and snugging the pan down jammed them instead. Sweeps went 7/7 -> 5/7 -> 2/7.
Welded pans plus stops keep the contents in, which is what the measurement needs.

The unknown is a **scanned asset**. Weighing a synthetic block against synthetic blocks tests
the rig; weighing an actual asset is the thing the project is for.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.data.assets import decimate, load_asset          # noqa: E402
from src.sim.mesh_scene import MeshProbeScene             # noqa: E402

GZ = 0.706
FPS, SUBSTEPS, NF = 24.0, 60, 56
BEAM_L, BEAM_W, BEAM_T = 0.62, 0.045, 0.012
# The pan has to SUIT ITS SUBJECT. At 90 mm radius with a 22 mm wall it was barely wider
# than the apple it was weighing (98 mm across), and a swinging pan simply tipped it out --
# both payloads ended on the table and the balance weighed its own empty pans.
# The pans are real scanned BOWLS, not fabricated cups. A flat pan with a wall lets a round
# subject roll to the rim and climb out -- the apple left the pan entirely at 12 degrees of
# tilt. Every fabricated alternative failed differently: a stepped well ejected the subject on
# contact with its shelf, a snug pan jammed it, and hanging pans swung it out. A bowl is
# concave all the way to its centre, so a round object settles at the bottom and stays there,
# which is the property that was missing and the reason real balance pans are dished.
PAN_ASSET, PAN_SCALE = ("rigid", "wooden_bowl_01"), 0.75
# How far the pivot sits ABOVE the beam assembly's centre of mass. This is the balance's
# sensitivity: at zero the beam is in neutral equilibrium and slams to a stop on any imbalance,
# carrying no magnitude information at all. Larger values give a stiffer, less sensitive
# instrument that settles at a small angle proportional to the load difference.
PIVOT_RISE = 0.030
ARM = 0.250
PIVOT_H = 0.20
COL_R, BASE_R, BASE_T = 0.016, 0.080, 0.016

# The unknown. A real scanned asset, weighed against calibrated references.
SUBJECT = ("rigid", "food_apple_01", 1.0, 0.182)    # cat, name, scale, TRUE mass (kg)
REFS = [0.05, 0.10, 0.15, 0.18, 0.22, 0.30, 0.50]


def pan_bowl():
    """The bowl mesh, scaled, with its base at z=0."""
    bowl = decimate(load_asset(*PAN_ASSET), 900).copy()
    bowl.apply_scale(PAN_SCALE)
    V = np.asarray(bowl.vertices)
    bowl.apply_translation([-V[:, 0].mean(), -V[:, 1].mean(), -V[:, 2].min()])
    return bowl


def pan_inner_z():
    """Height of the bowl's INNER floor above its base.

    A bowl is a shell: near its axis the lowest vertices are the OUTSIDE of the base and the
    highest are the inside surface a payload actually rests on. Seating a payload on the outer
    bottom buried it 12 mm inside the floor, and the contact then ejected it -- every subject
    ended on the table. Measure the inner surface instead of assuming the pan is a flat plate.
    """
    V = np.asarray(pan_bowl().vertices)
    r = np.linalg.norm(V[:, :2], axis=1)
    core = V[r < 0.15 * r.max()]
    return float(core[:, 2].max())


def beam_with_pans():
    """Beam plus two bowl-shaped pans, as ONE rigid body.

    Welded rather than hung: a suspended pan is a pendulum and its swing threw the payload out
    (sweeps went 7/7 -> 5/7 -> 2/7). `MeshProbeScene.add_link` remains available for suspended
    assemblies where that is not a problem.
    """
    beam = trimesh.creation.box(extents=(BEAM_L, BEAM_W, BEAM_T))
    parts = [beam]
    for sgn in (-1.0, +1.0):
        bowl = pan_bowl()
        bowl.apply_translation([sgn * ARM, 0.0, BEAM_T / 2])
        parts.append(bowl)
    return trimesh.util.concatenate(parts)


def stand_mesh():
    """Column and foot. It stops well below the pivot: the beam underside at tilt t and radius
    x sits at pivot_z - (BEAM_T/2)cos t - x*sin t, so a column reaching the level beam becomes
    a mechanical stop the moment it tilts."""
    top = PIVOT_H - 0.055
    col = trimesh.creation.cylinder(radius=COL_R, height=top, sections=24)
    col.apply_translation([0.0, 0.0, top / 2])
    base = trimesh.creation.cylinder(radius=BASE_R, height=BASE_T, sections=32)
    base.apply_translation([0.0, 0.0, BASE_T / 2])
    return trimesh.util.concatenate([col, base])


def subject_mesh():
    cat, name, sc, _ = SUBJECT
    tm = decimate(load_asset(cat, name), 500).copy()
    tm.apply_scale(sc)
    return tm


def ref_mesh(m):
    """A calibrated weight: a BLOCK, sized so its volume tracks its mass.

    Cylindrical references rolled. On a pan that swings even slightly a rolling weight walks
    inward and shortens its own moment arm -- measured 69 mm of travel, which at a 0.30 m arm
    is a 23% error in the very quantity being compared, and it biased the whole sweep toward
    the object side. A block stays where it is put.

    Volume tracks mass so the references look like what they weigh; a 0.5 kg reference
    identical in size to a 0.05 kg one would be a strange thing to put in front of a judge.
    """
    a = 0.038 * (m / 0.15) ** (1.0 / 3.0)
    return trimesh.creation.box(extents=(max(a, 0.026),) * 3)


def build(m_ref):
    z_beam = GZ + PIVOT_H          # world height of the PIVOT
    names = [beam_with_pans(), stand_mesh(), subject_mesh(), ref_mesh(m_ref)]
    masses = [0.45, 3.0, SUBJECT[3], m_ref]
    pos = [[0.0, 0.0, z_beam], [0.0, 0.0, GZ],
           [+ARM, 0.0, z_beam], [-ARM, 0.0, z_beam]]
    s = MeshProbeScene(names, pos, [[0., 0., 0.]] * len(names), masses=masses,
                       ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                       k=2500.0, cd=3000.0, mu=0.9, faces=500,
                       ground_mu=0.45, ground_cd=3000.0)
    p0 = s.pos0.copy()
    p0[1, 2] = s.rest_height(1)

    # Where the beam body's ORIGIN ends up. MeshProbeScene puts it at the assembly's vertex
    # mean -- which two large bowls move well above the beam box -- and the hinge holds a
    # point PIVOT_RISE above that origin at z_beam. Both offsets have to be undone to find
    # where a payload actually rests, rather than assuming the beam sits at the pivot.
    c_z = float(s.coms[0][2])
    p0[0, 2] = z_beam - PIVOT_RISE
    z_floor = p0[0, 2] + (BEAM_T / 2 + pan_inner_z() - c_z)
    for bi in (2, 3):                              # seat each payload in its bowl
        p0[bi, 2] = z_floor + s.sizes[bi] * 0.5 + 0.002
    s.pos0 = p0

    s.calibrate_stiffness()
    kk = s.k.numpy(); kk[0] = 2.0e5; kk[1] = 2.0e5; s.k.assign(kk)
    s.set_hinge(1, anchor=(0.0, 0.0, float(p0[1, 2])), axis=(0.0, 0.0, 1.0),
                damp_per_sec=1.0e6)
    # Stops at +/-12 deg keep the beam in a readable range; the pivot rise above the CoM is
    # what makes the angle between them mean something.
    s.set_hinge(0, anchor=(0.0, 0.0, z_beam), axis=(0.0, 1.0, 0.0),
                damp_per_sec=16.0, limit_deg=12.0,
                pivot_local=(0.0, 0.0, PIVOT_RISE))
    return s


def settled(ang, frac=0.25):
    n = max(int(len(ang) * frac), 1)
    return float(np.mean(ang[-n:]))


def run(m_ref, n_frames=NF):
    s = build(m_ref)
    s.rollout()
    Q = s.rotations(SUBSTEPS)[:, 0]
    ang = np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0)))
    return ang, s


def main():
    out = Path(os.environ.get("LAB", "outputs/scene/balance")); out.mkdir(parents=True, exist_ok=True)
    wp.init()
    true_m = SUBJECT[3]
    print(f"=== weighing {SUBJECT[1]} (true mass {true_m:.3f} kg) against calibrated references")
    print(f"  {'m_ref':>7}{'expect':>11}{'settled':>10}{'got':>11}{'ok':>5}")
    res = []
    with wp.ScopedDevice("cuda:0"):
        for mr in REFS:
            ang, s = run(mr)
            fin = settled(ang)
            exp = "obj down" if true_m > mr else ("ref down" if true_m < mr else "balance")
            got = "obj down" if fin > 0.8 else ("ref down" if fin < -0.8 else "balance")
            ok = (exp == got) or abs(true_m - mr) < 0.015
            res.append({"m_ref": mr, "m_obj": true_m, "tilt_deg": fin, "expect": exp,
                        "got": got, "ok": bool(ok), "ang": [float(a) for a in ang]})
            print(f"  {mr:>7.2f}{exp:>11}{fin:>10.2f}{got:>11}{str(ok):>5}")
    n_ok = sum(r["ok"] for r in res)
    # Bracket = the ADJACENT pair, in ascending reference mass, where the tilt changes sign.
    # Taking max(positive) and min(negative) over the whole sweep lets one stray case invert
    # the interval: a single lightest-reference outlier reported [0.18, 0.05].
    srt = sorted(res, key=lambda r: r["m_ref"])
    lo = hi = None
    for a, b in zip(srt, srt[1:]):
        if a["tilt_deg"] > 0.8 and b["tilt_deg"] < -0.8:
            lo, hi = a["m_ref"], b["m_ref"]
    print(f"  {n_ok}/{len(res)} correct")
    print(f"\n=== RECOVERY  true {true_m:.3f} kg   bracketed [{lo}, {hi}]")
    json.dump({"subject": SUBJECT[1], "m_obj": true_m, "sweep": res, "bracket": [lo, hi]},
              open(out / "balance.json", "w"), indent=1)
    print(f"  wrote {out}/balance.json")


if __name__ == "__main__":
    main()
