"""Write the whole scene back out as SimReady USD from the full-lab results.

Every parameter of every object carries its own status, not just the object:
    established   the seeds agreed, and the value is a measurement
    unverified    only one usable take, so there was no repeatability check
    not-established / no-take   the class prior is kept, and the asset says so

Densities are absolute numbers derived from the recovered mass RATIOS against the
reference object, anchored by one class prior for the reference. That anchoring is
explicit: relative mass is what a collision can actually reveal (M5), so the absolute
scale has to come from somewhere else and should be labelled as an assumption.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/write_full_simready.py    (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.data.assets import decimate, load_asset  # noqa: E402
from src.simready.usd_physics import coverage_report, write_physics  # noqa: E402
from src.sim.diff_collide_mesh import sphere_cover  # noqa: E402
from src.sim.probe_scene import ProbeScene  # noqa: E402

SCENE = REPO / "outputs" / "scene"
LAB = SCENE / "fulllab"
SUBSTEPS, K_CONTACT, PITCH = 60, 2500.0, 0.020
REF_DENSITY = 680.0        # class prior for the reference object, the one declared anchor

PRIOR = {
    "rubber_duck":  dict(density=180.0, dynamic_friction=0.70, restitution=0.30),
    "brass_pot":    dict(density=8400.0, dynamic_friction=0.35, restitution=0.10),
    "ceramic_vase": dict(density=2400.0, dynamic_friction=0.40, restitution=0.20),
    "wooden_bowl":  dict(density=700.0, dynamic_friction=0.50, restitution=0.30),
    "baseball":     dict(density=680.0, dynamic_friction=0.50, restitution=0.45),
    "apple":        dict(density=840.0, dynamic_friction=0.45, restitution=0.25),
    "book":         dict(density=800.0, dynamic_friction=0.55, restitution=0.05),
}


def restitution_from_cd(name, so, gz, cd):
    """Convert fitted contact damping into a real USD restitution coefficient by
    measuring the rebound the recovered simulation actually produces."""
    cat = so["asset"].split("/")[0]; asset = Path(so["asset"]).name
    tm = decimate(load_asset(cat, asset), 400).copy(); tm.apply_scale(so["scale"])
    centers, r = sphere_cover(tm, PITCH * so["scale"])
    vmean = np.asarray(tm.vertices).mean(0)
    c = [float(so["pos"][0] + vmean[0]), float(so["pos"][1] + vmean[1]),
         float(gz + r - centers[:, 2].min() + 0.18)]
    s = ProbeScene([f"{cat}/{asset}"], [c], [[0.0, 0.0, 0.0]], densities=(600.0,),
                   ground_z=gz, dt=1.0 / (24 * SUBSTEPS), n_steps=55 * SUBSTEPS,
                   k=K_CONTACT, cd=float(cd), mu=0.4, mesh_scale=[so["scale"]], pitch=PITCH)
    s.rollout(); P = s.positions(4)
    z = P[:, 0, 2]; vz = np.diff(z)
    hit = int(np.argmin(z[:len(z) * 3 // 4]))
    if hit < 2 or hit + 4 >= len(vz):
        return None
    down = abs(vz[max(hit - 2, 0):hit + 1].min()); up = abs(vz[hit:hit + 4].max())
    return float(np.clip(up / max(down, 1e-6), 0.0, 0.98)) if down > 1e-6 else None


def main():
    scene = json.loads((SCENE / "scene.json").read_text())
    rec = json.loads((LAB / "recovered.json").read_text())
    gz = scene["ground_z"]
    wp.init()
    values = {}
    with wp.ScopedDevice("cuda:0"):
        for name, so in scene["objects"].items():
            v = dict(PRIOR.get(name, dict(density=800.0, dynamic_friction=0.5, restitution=0.3)))
            prov = {k: "default:class-prior" for k in ("density", "dynamic_friction", "restitution")}
            conf = {k: 0.0 for k in prov}
            r = rec.get(name, {})

            cdr = r.get("cd", {})
            if cdr.get("value") is not None:
                e = restitution_from_cd(name, so, gz, cdr["value"])
                if e is not None:
                    v["restitution"] = e
                    n = cdr.get("n", 1); ag = cdr.get("agree")
                    prov["restitution"] = (f"recovered:drop:{n}takes"
                                           + (f":agree{ag:.1f}x" if ag else ":unverified"))
                    conf["restitution"] = 0.35 if ag is None else float(np.clip(1.0 - (ag - 1) / 3.0, 0.1, 0.95))

            mur = r.get("mu", {})
            # A value sitting on the edge of the search grid is a bound, not a measurement:
            # the optimum is somewhere outside what we scanned. The apple came back at
            # exactly 1.100, the grid maximum, which is also physically odd (mu>1 is very
            # grippy for fruit on wood) — so it is downgraded rather than written.
            if mur.get("value") is not None and abs(mur["value"] - 1.1) < 1e-6:
                mur = {"value": None, "why": "railed at the top of the searched range"}
            if mur.get("value") is not None:
                v["dynamic_friction"] = float(mur["value"])
                n = mur.get("n", 1); ag = mur.get("agree")
                prov["dynamic_friction"] = (f"recovered:slide:{n}takes"
                                            + (f":agree{ag:.1f}x" if ag else ":unverified"))
                conf["dynamic_friction"] = 0.35 if ag is None else float(np.clip(1.0 - (ag - 1) / 3.0, 0.1, 0.95))

            rar = r.get("ratio", {})
            # same rule as friction: a value sitting on the edge of the searched grid is a
            # bound, not a measurement (the duck came back at exactly 0.200, the minimum)
            if rar.get("value") is not None and (abs(rar["value"] - 0.2) < 1e-6 or
                                                 abs(rar["value"] - 5.0) < 1e-6):
                rar = {"value": None, "why": "railed at the edge of the searched range"}
            if rar.get("value") is not None:
                # ratio is partner/subject density with the reference as partner
                v["density"] = float(REF_DENSITY / max(rar["value"], 1e-6))
                n = rar.get("n", 1); ag = rar.get("agree")
                prov["density"] = (f"recovered:collide:{n}takes"
                                   + (f":agree{ag:.1f}x" if ag else ":unverified")
                                   + f":anchored-to-{REF_DENSITY:.0f}")
                conf["density"] = 0.30 if ag is None else float(np.clip(0.9 - (ag - 1) / 3.0, 0.1, 0.9))

            v["provenance"] = prov; v["confidence"] = conf
            values[name] = v
            got = [k for k, s in prov.items() if s.startswith("recovered")]
            print(f"  {name:13s} measured: {', '.join(got) if got else '-'}")

    out = SCENE / "scene_simready.usda"
    written = write_physics(SCENE / "scene_geom.usdc", out, values)
    (SCENE / "values.json").write_text(json.dumps(values, indent=2))
    nvals = sum(1 for v in values.values() for s in v["provenance"].values() if s.startswith("recovered"))
    nobj = sum(1 for v in values.values() if any(s.startswith("recovered") for s in v["provenance"].values()))
    print(f"\nwrote {out} ({len(written)} objects)")
    print(f"COVERAGE: {nvals} measured values across {nobj}/{len(values)} objects\n")
    print(coverage_report(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
