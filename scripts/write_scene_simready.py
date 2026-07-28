"""Write the scene back out as SimReady USD, using whatever was actually recovered.

Values that were recovered from a generated probe video are written with
provenance "recovered:<probe>:<seed>". Values no probe ever pinned down keep a class
prior and are written with provenance "default:class-prior" and confidence 0 — an asset
that admits which of its numbers are measurements.

The fitted contact-damping is converted into an honest USD restitution COEFFICIENT by
running the recovered simulation and measuring the actual rebound-to-impact speed ratio,
rather than inventing a mapping.

Run: CUDA_VISIBLE_DEVICES=<g> python scripts/write_scene_simready.py    (warp env)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

from src.data.assets import load_asset  # noqa: E402
from src.simready.usd_physics import coverage_report, write_physics  # noqa: E402
from src.sim.probe_scene import ProbeScene  # noqa: E402

SCENE = REPO / "outputs" / "scene"
PROBES = SCENE / "probes"
K_CONTACT, SUBSTEPS = 2500.0, 80

# class priors used when nothing was measured (kg/m^3 and dimensionless)
PRIOR = {
    "rubber_duck":  dict(density=180.0, dynamic_friction=0.70, restitution=0.30),
    "brass_pot":    dict(density=8400.0, dynamic_friction=0.35, restitution=0.10),
    "ceramic_vase": dict(density=2400.0, dynamic_friction=0.40, restitution=0.20),
    "wooden_bowl":  dict(density=700.0, dynamic_friction=0.50, restitution=0.30),
    "baseball":     dict(density=680.0, dynamic_friction=0.50, restitution=0.45),
    "apple":        dict(density=840.0, dynamic_friction=0.45, restitution=0.25),
    "book":         dict(density=800.0, dynamic_friction=0.55, restitution=0.05),
}


def measure_restitution(name, so, drop_h, gz, cd):
    """Run the recovered sim and read off rebound speed / impact speed."""
    cat = so["asset"].split("/")[0]
    asset = Path(so["asset"]).name
    tm = load_asset(cat, asset)
    vmean = np.asarray(tm.vertices).mean(0) * so["scale"]
    c = np.array(so["pos"], float) + vmean
    sc = ProbeScene([f"{cat}/{asset}"], [[c[0], c[1], c[2] + drop_h]], [[0.0, 0.0, 0.0]],
                    densities=(600.0,), ground_z=gz, dt=1.0 / (24 * SUBSTEPS),
                    n_steps=60 * SUBSTEPS, k=K_CONTACT, cd=float(cd), mu=0.4,
                    mesh_scale=[so["scale"]], pitch=0.020)
    sc.rollout()
    P = sc.positions(4)
    z = P[:, 0, 2]
    vz = np.diff(z)
    hit = int(np.argmin(z[:len(z) * 3 // 4]))          # first deepest point = impact
    down = abs(vz[max(hit - 2, 0):hit + 1].min()) if hit > 0 else 0.0
    up = abs(vz[hit:hit + 4].max()) if hit + 4 < len(vz) else 0.0
    if down < 1e-6:
        return None
    return float(np.clip(up / down, 0.0, 0.98))


def main():
    scene = json.loads((SCENE / "scene.json").read_text())
    probes = json.loads((PROBES / "probes.json").read_text())
    rec = json.loads((PROBES / "recovered.json").read_text()) if (PROBES / "recovered.json").exists() else {}
    gz = scene["ground_z"]

    wp.init()
    values = {}
    with wp.ScopedDevice("cuda:0"):
        for name, so in scene["objects"].items():
            v = dict(PRIOR.get(name, dict(density=800.0, dynamic_friction=0.5, restitution=0.3)))
            prov = {k: "default:class-prior" for k in ("density", "dynamic_friction", "restitution")}
            conf = {k: 0.0 for k in prov}
            r = rec.get(name)
            if r and r.get("identified"):
                e = measure_restitution(name, so, probes["drop_h"], gz, r["cd"])
                if e is not None:
                    v["restitution"] = e
                    prov["restitution"] = f"recovered:drop:seed{r['seed']}:{r.get('loss','pixel')}-loss"
                    # confidence from how tightly the value is pinned: a narrow interval
                    # of damping values that fit the signature means a real measurement
                    conf["restitution"] = float(np.clip(1.0 - r.get("frac", 0.5), 0.05, 0.95))
                    iv = r.get("interval", [r["cd"], r["cd"]])
                    print(f"  {name:13s} restitution {e:.3f}  (damping {r['cd']:.1f}, "
                          f"pinned to [{iv[0]:.0f}-{iv[1]:.0f}], confidence {conf['restitution']:.2f})")
                else:
                    print(f"  {name:13s} identified but the rebound could not be measured "
                          f"-> class prior kept")
            elif r:
                print(f"  {name:13s} not recovered: {r.get('why','-')}  -> class prior kept")
            else:
                print(f"  {name:13s} no probe run -> class prior kept")
            v["provenance"] = prov; v["confidence"] = conf
            values[name] = v

    out = SCENE / "scene_simready.usda"
    written = write_physics(SCENE / "scene_geom.usdc", out, values)
    (SCENE / "values.json").write_text(json.dumps(values, indent=2))
    n_rec = sum(1 for v in values.values()
                if any(s.startswith("recovered") for s in v["provenance"].values()))
    print(f"\nwrote {out}  ({len(written)} objects)")
    print(f"COVERAGE: {n_rec}/{len(values)} objects carry at least one measured value\n")
    print(coverage_report(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
