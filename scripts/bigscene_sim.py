"""A LARGER scene: 14 objects on one widened table, one continuous tilt ramp.

Everything before this fitted two objects (brass_pot, wooden_bowl) because the probes were
paired to specific geometry one at a time. But the tilt ramp is a *scene-level* experiment:
tilt the table and every object declares itself at once -- low-friction things slide first,
top-heavy things topple, spheres roll. One rollout, one render, 14 readings.

Uses the ramp fixes validated in docs/SCALE_RANGE.md:
  - ONE continuous rollout with per-step gravity (no per-angle scene restarts)
  - contact stiffness k ~ s^2 so penetration is a fixed fraction of object size
  - the sweep run in dimensionless time, dt ~ sqrt(s)

Writes LAB/scene_poses.json for scripts/blender_render_scene.py.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")
from src.data.assets import decimate, load_asset          # noqa: E402
from src.sim.diff_collide_mesh import sphere_cover        # noqa: E402
from src.sim.probe_scene import ProbeScene                # noqa: E402
from src.sim.tilt_probe import ramp_gravity_seq, onset_angle  # noqa: E402

LAB = Path(os.environ.get("LAB", "outputs/scene/bigscene"))
LAB.mkdir(parents=True, exist_ok=True)
GZ = 0.706
FPS, SUBSTEPS, NF, SETTLE = 24.0, 60, 60, 12
DEG0, DEG1 = 0.0, 26.0
K_CONTACT, PITCH, DENSITY = 2500.0, 0.020, 600.0
TABLE_W, TABLE_D = 1.134 * 1.40, 0.706 * 1.40      # matches TABLE_SX/SY in the render
MARGIN, GAP = 0.06, 0.045

# per-object friction: what we would be fitting. Spread across a plausible range so the
# scene shows a RANGE of behaviours rather than 14 objects doing the same thing.
MU = {
    "wooden_bowl_01": 0.22, "Pony_C_Clamp_1440": 0.18, "Weisshai_Great_White_Shark": 0.30,
    "brass_pot_01": 0.26, "Threshold_Porcelain_Teapot_White": 0.24,
    "cardboard_box_01": 0.45, "Poppin_File_Sorter_Blue": 0.38,
    "ceramic_vase_01": 0.35, "book_encyclopedia_set_01": 0.40,
    "rubber_duck_toy": 0.55, "Great_Dinos_Triceratops_Toy": 0.42,
    "Schleich_Lion_Action_Figure": 0.42, "baseball_01": 0.32, "food_apple_01": 0.34,
}


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def layout(assets):
    """Shelf-pack by footprint depth. Rows run along +x, which is the tilt-down direction."""
    items = sorted(assets.items(), key=lambda kv: -kv[1]["ext"][1])
    rows, cur, curw, curd = [], [], 0.0, 0.0
    for k, v in items:
        w = v["ext"][0] + GAP
        if curw + w > TABLE_W - 2 * MARGIN and cur:
            rows.append((cur, curd)); cur, curw, curd = [], 0.0, 0.0
        cur.append((k, v)); curw += w; curd = max(curd, v["ext"][1])
    if cur:
        rows.append((cur, curd))
    totd = sum(d for _, d in rows) + GAP * (len(rows) - 1)
    out, y = {}, -totd / 2
    for row, rd in rows:
        roww = sum(v["ext"][0] + GAP for _, v in row) - GAP
        x = -roww / 2
        for k, v in row:
            out[k] = [float(x + v["ext"][0] / 2), float(y + rd / 2)]
            x += v["ext"][0] + GAP
        y += rd + GAP
    return out, totd


def main():
    A = json.load(open(sys.argv[1]))
    wp.init()
    xy, totd = layout(A)
    print(f"layout: {len(A)} objects, depth {totd:.3f} m on a {TABLE_W:.2f}x{TABLE_D:.2f} table")
    assert totd <= TABLE_D - 2 * MARGIN, f"overflows depth by {totd - (TABLE_D - 2*MARGIN):.3f}"

    names, pos, scales, mus, keys = [], [], [], [], []
    for k, v in A.items():
        s = v["scale"]
        tm = decimate(load_asset(v["cat"], k), 400).copy(); tm.apply_scale(s)
        ctr, rad = sphere_cover(tm, PITCH * s)
        z = GZ + rad - float(ctr[:, 2].min()) + 0.002 * s
        names.append(f"{v['cat']}/{k}"); pos.append([xy[k][0], xy[k][1], z])
        scales.append(s); mus.append(MU.get(k, 0.35)); keys.append(k)

    # ONE scene, per-body friction: the objects genuinely collide with each other, and
    # each still slides at its own angle. Simulating them separately and compositing would
    # let them pass through one another, which is not a scene.
    onsets = {}
    with wp.ScopedDevice("cuda:0"):
        sn = ProbeScene(names, pos, [[0., 0., 0.]] * len(names),
                        densities=tuple([DENSITY] * len(names)), ground_z=GZ,
                        dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                        k=K_CONTACT, cd=3000.0, mu=mus,
                        mesh_scale=scales, pitch=PITCH)
        sn.gravity_seq = ramp_gravity_seq(DEG0, DEG1, NF, SUBSTEPS, SETTLE)
        sn.rollout()
        P, Q = sn.positions(SUBSTEPS)[:NF], sn.rotations(SUBSTEPS)[:NF]
        assert np.isfinite(P).all(), "rollout diverged"
        print(f"one scene: {len(names)} bodies, {sn.nS} spheres, mu per body")
        for i, k in enumerate(keys):
            size = float(max(load_asset(A[k]["cat"], k).extents) * scales[i])
            o = onset_angle(P[:, i, :2], DEG0, DEG1, NF, SETTLE, size)
            onsets[k] = {"mu": float(mus[i]), "onset_deg": o,
                         "true_slip_deg": float(np.rad2deg(np.arctan(mus[i]))),
                         "probe": A[k]["probe"],
                         "travel_cm": float(np.linalg.norm(P[-1, i, :2] - P[SETTLE, i, :2]) * 100)}
        for k in sorted(onsets, key=lambda z: onsets[z]["mu"]):
            d = onsets[k]
            print(f"  {k:<38} mu={d['mu']:.2f}  true {d['true_slip_deg']:>5.1f}d  "
                  f"onset {('%.1fd' % d['onset_deg']) if d['onset_deg'] else '  none':>7}  "
                  f"travel {d['travel_cm']:>6.1f} cm  [{d['probe']}]")

    ramp = np.concatenate([np.full(SETTLE, DEG0),
                           np.linspace(DEG0, DEG1, NF - SETTLE)])
    scenes = {}
    for t in range(NF):
        scenes.setdefault("bigscene_tilt", {"tilt_deg": [], "objects": {}})
    S = scenes["bigscene_tilt"]
    S["tilt_deg"] = [float(a) for a in ramp]
    # FRAME CONVENTION. The simulator keeps the ground flat and rotates GRAVITY; the
    # renderer keeps gravity down and rotates the TABLE. To show the same physics, every
    # pose must be carried into the tilted-table frame: rotate about Y through the pivot
    # on the table surface, by the same angle, matching Matrix.Rotation(th, 'Y') in
    # blender_render_scene.py (+x tilts down). Writing raw simulator poses instead leaves
    # objects on a flat plane while the table rotates out from under them.
    PIVOT = np.array([-0.05, -0.06, GZ])
    for i, k in enumerate(keys):
        seq = []
        for t in range(NF):
            th = np.deg2rad(ramp[t])
            Ry = np.array([[np.cos(th), 0.0, np.sin(th)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(th), 0.0, np.cos(th)]])
            loc = PIVOT + Ry @ (np.asarray(P[t, i], float) - PIVOT)
            mat = Ry @ quat_to_mat(Q[t, i])
            seq.append({"loc": [float(v) for v in loc],
                        "mat": [[float(v) for v in r] for r in mat]})
        S["objects"][k] = seq
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "onsets.json").write_text(json.dumps(onsets, indent=1))
    # scene.json is what the Blender script reads: it needs asset path, rest pose, scale.
    (LAB / "scene.json").write_text(json.dumps({
        "ground_z": GZ,
        "objects": {k: {"asset": f"{A[k]['cat']}/{k}", "scale": A[k]["scale"],
                        "rot_z": 0.0, "pos": pos[i]} for i, k in enumerate(keys)}}, indent=1))
    print(f"wrote {LAB}/scene_poses.json  ({NF} frames, {len(keys)} objects)")


if __name__ == "__main__":
    main()
