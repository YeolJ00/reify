"""Solve three different experiments together for one set of physical parameters.

    drop      the baseball falls              -> restitution
    slide     it travels across the table     -> friction
    collide   it strikes the apple            -> mass ratio

The three clips share ONE material, so they are fitted as a single system:

    shared      cd (restitution), mu (friction), rho_target/rho_mover (mass ratio)
    per-clip    the launch velocity of the slide and the collide (Cosmos chose it, we
                did not), and the drop's effective height (it often drops only part way)

Fitting separately would mean holding the others at a guess for each one, and a wrong
guess biases whatever does get reported. Fitting jointly also exposes trade-offs: if two
parameters can compensate for each other, the joint fit shows it instead of hiding it
behind a fixed value.

Three passes (env split is because src.render.camera imports warp, absent in `video`):
    SEED_ONLY=1  warp env   where each object is in frame 0
    TRACK_ONLY=1 video env  CoTracker
    (default)    warp env   the joint fit
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LAB = REPO / "outputs" / "scene" / "lab"
SUBSTEPS, K_CONTACT = 60, 2500.0
EXPS = ["drop", "slide", "collide"]


def _obj_geom(so, load_asset, decimate=None, ground_z=None, pitch=0.020):
    """(body centre in world, mesh key, scale) for a placed scene object.

    The centre must be derived from the SAME sphere cover the simulator builds, not from
    the mesh bounds: the cover is a voxelised shell of finite-radius spheres centred on
    the mesh's vertex mean, so the body has to sit one sphere radius above where the
    geometry alone suggests, and the voxel grid shifts the lowest point again. Deriving it
    any other way left objects buried in the table — they popped up 6 cm on frame 1 and
    every signature computed downstream was measuring that artefact.
    """
    from src.sim.diff_collide_mesh import sphere_cover
    cat = so["asset"].split("/")[0]; asset = Path(so["asset"]).name
    scale = so["scale"]
    tm = load_asset(cat, asset)
    if decimate is not None:
        tm = decimate(tm, 400)
    tm = tm.copy(); tm.apply_scale(scale)
    centers, r = sphere_cover(tm, pitch * scale)
    vmean = np.asarray(tm.vertices).mean(0)          # already scaled
    pos = np.array(so["pos"], float)
    c = np.array([pos[0] + vmean[0], pos[1] + vmean[1], 0.0])
    # rest height: lowest sphere of the cover just touches the table
    c[2] = (ground_z if ground_z is not None else 0.0) + r - float(centers[:, 2].min())
    return c, f"{cat}/{asset}", scale


def write_seeds():
    from src.data.assets import decimate, load_asset
    from src.render.camera import Camera
    cfg = json.loads((LAB / "lab.json").read_text())
    cam = Camera(cfg["camera"])
    seeds = {}
    for ename, e in cfg["experiments"].items():
        s = {}
        for role, key in (("mover", e["mover"]), ("target", e["target"])):
            so = dict(cfg["assets"][key])
            so["pos"] = e[f"{role}_pos"]
            c, _mk, _sc = _obj_geom(so, load_asset, decimate, cfg['ground_z'])
            c[2] += float(e.get('mover_lift', 0.0)) if role == 'mover' else 0.0
            uv, dep = cam.project(np.array([c]))
            size_m = min(so["size_cm"][0], so["size_cm"][1]) / 100.0
            s[role] = {"u": float(uv[0][0]), "v": float(uv[0][1]),
                       "r": float(0.40 * cam.fx * size_m / max(float(dep[0]), 1e-6)),
                       "size_px": float(cam.fx * size_m / max(float(dep[0]), 1e-6))}
        seeds[ename] = s
        print(f"  {ename:8s} mover@({s['mover']['u']:.0f},{s['mover']['v']:.0f}) "
              f"target@({s['target']['u']:.0f},{s['target']['v']:.0f})")
    (LAB / "seeds.json").write_text(json.dumps(seeds, indent=2))
    return 0


def track():
    import glob
    from src.track.cotracker import seed_in_mask, track_points
    seeds = json.loads((LAB / "seeds.json").read_text())
    for ename in EXPS:
        for f in sorted(glob.glob(str(LAB / f"vid_{ename}_seed*.npz"))):
            sd = Path(f).stem.split("seed")[1]
            dst = LAB / f"trk_{ename}_seed{sd}.npz"
            if dst.exists():
                continue
            fr = np.load(f)["frames"]; H, W, _ = fr[0].shape
            yy, xx = np.mgrid[0:H, 0:W]
            out = {}
            for role in ("mover", "target"):
                s = seeds[ename][role]
                mask = ((xx - s["u"]) ** 2 + (yy - s["v"]) ** 2) < s["r"] ** 2
                q = seed_in_mask(mask, n=36, seed=0)
                tr, vis = track_points(fr, q, device="cuda")
                cen = np.array([np.median(tr[t, vis[t]] if vis[t].mean() > 0.3 else tr[t], axis=0)
                                for t in range(len(fr))])
                out[f"{role}_cen"] = cen; out[f"{role}_vis"] = vis
            np.savez(dst, **out)
            print(f"tracked {ename} seed{sd}: mover {out['mover_vis'].mean()*100:.0f}% "
                  f"target {out['target_vis'].mean()*100:.0f}%", flush=True)
    return 0


def main():
    if os.environ.get("SEED_ONLY"):
        return write_seeds()
    if os.environ.get("TRACK_ONLY"):
        return track()

    import warp as wp
    from src.data.assets import decimate, load_asset
    from src.motion.signature import (collide_distance, collide_signature, drop_signature,
                                      signature_distance, slide_distance, slide_signature)
    from src.render.camera import Camera
    from src.sim.probe_scene import ProbeScene

    cfg = json.loads((LAB / "lab.json").read_text())
    seeds = json.loads((LAB / "seeds.json").read_text())
    cam = Camera(cfg["camera"]); GZ = cfg["ground_z"]

    # ---- observations: best-tracked take per experiment ----
    obs = {}
    for ename in EXPS:
        best = None
        for t in sorted(LAB.glob(f"trk_{ename}_seed*.npz")):
            d = np.load(t)
            q = float(d["mover_vis"].mean() + (d["target_vis"].mean() if ename == "collide" else 0))
            if best is None or q > best[0]:
                best = (q, d, t.stem.split("seed")[1])
        if best is None:
            print(f"{ename}: no tracks"); return 1
        _q, d, sd = best
        sp = seeds[ename]["mover"]["size_px"]
        if ename == "drop":
            sig = drop_signature(d["mover_cen"][:, 1], d["mover_cen"][:, 0])
        elif ename == "slide":
            sig = slide_signature(d["mover_cen"], sp)
        else:
            sig = collide_signature(d["mover_cen"], d["target_cen"], sp)
        obs[ename] = {"sig": sig, "cen": d["mover_cen"], "tgt": d.get("target_cen"),
                      "seed": sd, "size_px": sp, "n": len(d["mover_cen"])}
        print(f"  {ename:8s} seed{sd}: {sig}")
        if sig is None:
            print(f"    ^ unusable — that experiment contributes nothing")

    mv = dict(cfg["assets"][cfg["mover"]]); tg = dict(cfg["assets"][cfg["target"]])
    wp.init()

    def run(ename, th, v0):
        e = cfg["experiments"][ename]
        m_so = dict(mv); m_so["pos"] = e["mover_pos"]
        t_so = dict(tg); t_so["pos"] = e["target_pos"]
        mc, mk, msc = _obj_geom(m_so, load_asset, decimate, GZ)
        mc[2] += float(e.get('mover_lift', 0.0))     # the drop's lift
        tc, tk, tsc = _obj_geom(t_so, load_asset, decimate, GZ)
        nf = obs[ename]["n"]
        if ename == "collide":
            sc = ProbeScene([mk, tk], [list(mc), list(tc)], [[v0[0], v0[1], 0.0], [0, 0, 0]],
                            densities=(600.0, 600.0 * th["ratio"]), ground_z=GZ,
                            dt=1.0 / (24 * SUBSTEPS), n_steps=nf * SUBSTEPS, k=K_CONTACT,
                            cd=th["cd"], mu=th["mu"], mesh_scale=[msc, tsc], pitch=0.020)
        else:
            sc = ProbeScene([mk], [list(mc)], [[v0[0], v0[1], 0.0]], densities=(600.0,),
                            ground_z=GZ, dt=1.0 / (24 * SUBSTEPS), n_steps=nf * SUBSTEPS,
                            k=K_CONTACT, cd=th["cd"], mu=th["mu"], mesh_scale=[msc], pitch=0.020)
        sc.rollout(); P = sc.positions(SUBSTEPS)
        if not np.isfinite(P).all():
            return None
        return P[:nf]

    def cost(z):
        th = {"cd": float(np.clip(np.exp(z[0]), 0.5, 400.0)),
              "mu": float(np.clip(z[1], 0.02, 1.2)),
              "ratio": float(np.clip(np.exp(z[2]), 0.15, 8.0))}
        v = {"drop": (0.0, 0.0), "slide": (z[3], z[4]), "collide": (z[5], z[6])}
        tot, n = 0.0, 0
        for ename in EXPS:
            if obs[ename]["sig"] is None:
                continue
            P = run(ename, th, v[ename])
            if P is None:
                tot += 1.0; n += 1; continue
            uv, _ = cam.project(P[:, 0])
            sp = obs[ename]["size_px"]
            if ename == "drop":
                d = signature_distance(drop_signature(uv[:, 1], uv[:, 0]), obs[ename]["sig"])
            elif ename == "slide":
                d = slide_distance(slide_signature(uv, sp), obs[ename]["sig"])
            else:
                uvt, _ = cam.project(P[:, 1])
                d = collide_distance(collide_signature(uv, uvt, sp), obs[ename]["sig"])
            tot += min(d, 1.0); n += 1
        return tot / max(n, 1), th

    # ---------------- SEQUENTIAL mode (dependency-graph order) ----------------
    # The parameters are not symmetric: a drop needs nothing else to reveal restitution,
    # a slide needs nothing else to reveal friction, but a collision involves BOTH of
    # those before any mass information is left over. So there is a natural order
    #     restitution  <- drop
    #     friction     <- slide
    #     mass ratio   <- collide, holding the first two fixed
    # Solving in that order is easier to debug and degrades gracefully: if the collision
    # fails you still keep the first two, instead of one opaque fit that half-worked.
    if os.environ.get("SEQ"):
        fixed = {"cd": 20.0, "mu": 0.4, "ratio": 1.0}
        nuis = {"drop": (0.0, 0.0), "slide": (0.6, 0.0), "collide": (0.9, 0.0)}
        stages = [("cd", "drop", np.geomspace(0.5, 400.0, 11)),
                  ("mu", "slide", np.linspace(0.05, 1.2, 11)),
                  ("ratio", "collide", np.geomspace(0.15, 8.0, 11))]
        print("\nSEQUENTIAL fit, in dependency order")
        for key, ename, grid in stages:
            if obs[ename]["sig"] is None:
                print(f"  {key:6s} <- {ename:8s}: experiment unusable, left at prior {fixed[key]}")
                continue
            errs = []
            for g in grid:
                th = dict(fixed); th[key] = float(g)
                b = 1e9
                for vx in ([0.0] if ename == "drop" else np.linspace(0.25, 1.6, 5)):
                    P = run(ename, th, (vx, 0.0))
                    if P is None:
                        continue
                    uv, _ = cam.project(P[:, 0]); sp = obs[ename]["size_px"]
                    if ename == "drop":
                        dd = signature_distance(drop_signature(uv[:, 1], uv[:, 0]), obs[ename]["sig"])
                    elif ename == "slide":
                        dd = slide_distance(slide_signature(uv, sp), obs[ename]["sig"])
                    else:
                        uvt, _ = cam.project(P[:, 1])
                        dd = collide_distance(collide_signature(uv, uvt, sp), obs[ename]["sig"])
                    b = min(b, dd)
                errs.append(b)
            errs = np.array(errs); i = int(np.argmin(errs))
            span = float(errs.max() - errs.min())
            near = errs <= errs[i] + 0.03
            frac = float(near.sum() / len(grid))
            det = span > 0.02 and frac < 0.5 and i not in (0, len(grid) - 1)
            fixed[key] = float(grid[i])
            print(f"  {key:6s} <- {ename:8s}: {grid[i]:8.3f}  fit {errs[i]:.4f}  "
                  f"{'DETERMINED' if det else 'not constrained'}  "
                  f"(cost range {span:.3f}, {100*frac:.0f}% of the range fits)")
        print(f"\n  sequential result: cd={fixed['cd']:.2f} mu={fixed['mu']:.3f} ratio={fixed['ratio']:.3f}")
        np.savez(LAB / "seq_fit.npz", theta=json.dumps(fixed))
        return 0

    rng = np.random.default_rng(0)
    m = np.array([np.log(20.0), 0.4, 0.0, 0.6, 0.0, 0.9, 0.0])
    s = np.array([1.3, 0.3, 0.7, 0.5, 0.25, 0.6, 0.25])
    best = (1e9, m.copy())
    POP, IT = 20, 9
    print(f"\njoint fit: 3 shared parameters + 2 launch velocities, over "
          f"{sum(1 for e in EXPS if obs[e]['sig'] is not None)} experiments")
    for it in range(IT):
        cand = m + s * rng.standard_normal((POP, len(m)))
        sc_ = np.array([cost(c)[0] for c in cand])
        o = np.argsort(sc_); el = cand[o[:6]]
        if sc_[o[0]] < best[0]:
            best = (float(sc_[o[0]]), cand[o[0]].copy())
        m, s = el.mean(0), np.maximum(el.std(0), np.array([.06, .02, .06, .04, .03, .04, .03]))
        _, th = cost(m)
        print(f"  it {it}: cost {best[0]:.4f} | cd {th['cd']:6.2f} mu {th['mu']:.3f} "
              f"mass ratio {th['ratio']:.3f}")
    c, th = cost(best[1])
    # WHICH parameters did this experiment set actually determine? Perturb each around
    # the optimum: if the joint cost barely notices, nothing in these clips constrains it
    # and reporting a number would be inventing one. (With the collision rejected, the
    # mass ratio should come out unconstrained — that is the correct answer, not a value.)
    SENS = {}
    for idx, key, delta in ((0, "cd", 0.9), (1, "mu", 0.28), (2, "ratio", 0.9)):
        lo = best[1].copy(); hi = best[1].copy()
        lo[idx] -= delta; hi[idx] += delta
        SENS[key] = max(cost(lo)[0], cost(hi)[0]) - c
    tolm = 0.02
    print(f"\nJOINTLY RECOVERED (baseball vs apple), joint cost {c:.4f}")
    for key, lab, val in (("cd", "restitution damping", th["cd"]),
                          ("mu", "friction mu        ", th["mu"]),
                          ("ratio", "mass ratio apple/ball", th["ratio"])):
        det = SENS[key] > tolm
        print(f"  {lab} = {val:8.3f}   {'DETERMINED' if det else 'NOT constrained by these experiments'}"
              f"   (cost moves {SENS[key]:.4f} when perturbed)")
    print("\n  experiments used: " + ", ".join(e for e in EXPS if obs[e]["sig"] is not None))
    print("  experiments rejected: " + (", ".join(e for e in EXPS if obs[e]["sig"] is None) or "none"))
    np.savez(LAB / "joint_fit.npz", theta=json.dumps(th), z=best[1], cost=c,
             sensitivity=json.dumps(SENS),
             used=json.dumps([e for e in EXPS if obs[e]["sig"] is not None]),
             obs=json.dumps({k: (v["sig"] if v["sig"] else None) for k, v in obs.items()}))
    print(f"wrote {LAB}/joint_fit.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
