"""Write recovered physical values back onto the scene as SimReady USD attributes.

This is the deliverable: the same scene the artist authored, handed back with physics
filled in, so it can be dropped straight into a simulator.

Per movable object we apply
    UsdPhysics.RigidBodyAPI   - it is a dynamic body
    UsdPhysics.CollisionAPI   - it collides
    UsdPhysics.MassAPI        - density (mass falls out of density x volume)
and bind a physics material carrying
    dynamicFriction / staticFriction / restitution
Static geometry (the table, the floor) gets CollisionAPI + a material but no rigid body.

Every value also carries PROVENANCE as custom attributes: was it recovered from video,
or left at a class default because no probe ever exercised it, and how confident are we.
An honest asset says which of its numbers are real — see `coverage_report`.
"""
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

PHYSICS_PURPOSE = getattr(UsdShade.Tokens, "physics", "physics")


def _material(stage, name, vals):
    path = f"/root/PhysicsMaterials/{name}_physmat"
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateDynamicFrictionAttr(float(vals["dynamic_friction"]))
    api.CreateStaticFrictionAttr(float(vals.get("static_friction", vals["dynamic_friction"] * 1.2)))
    api.CreateRestitutionAttr(float(vals["restitution"]))
    if "density" in vals:
        api.CreateDensityAttr(float(vals["density"]))
    return mat


def _provenance(prim, vals):
    """Record where each number came from, so the asset is self-describing."""
    for key, src in vals.get("provenance", {}).items():
        prim.CreateAttribute(f"simready:provenance:{key}", Sdf.ValueTypeNames.String).Set(str(src))
    for key, c in vals.get("confidence", {}).items():
        prim.CreateAttribute(f"simready:confidence:{key}", Sdf.ValueTypeNames.Float).Set(float(c))


def write_physics(geom_usd, out_usd, values, static_prims=("wooden_table_02", "Plane"),
                  gravity=9.81):
    """values: {object_name: {density, dynamic_friction, restitution, provenance, confidence}}"""
    stage = Usd.Stage.Open(str(geom_usd))
    if stage is None:
        raise RuntimeError(f"cannot open {geom_usd}")

    scene = UsdPhysics.Scene.Define(stage, "/root/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(gravity))

    written = []
    for name, vals in values.items():
        prim = stage.GetPrimAtPath(f"/root/{name}")
        if not prim or not prim.IsValid():
            print(f"  skip {name}: not in stage")
            continue
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        if "density" in vals:
            mass.CreateDensityAttr(float(vals["density"]))
        for child in [prim] + list(prim.GetChildren()):
            if UsdGeom.Mesh(child):
                UsdPhysics.CollisionAPI.Apply(child)
        mat = _material(stage, name, vals)
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                               PHYSICS_PURPOSE)
        _provenance(prim, vals)
        written.append(name)

    for name in static_prims:
        prim = stage.GetPrimAtPath(f"/root/{name}")
        if prim and prim.IsValid():
            for child in [prim] + list(prim.GetChildren()):
                if UsdGeom.Mesh(child):
                    UsdPhysics.CollisionAPI.Apply(child)

    Path(out_usd).parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    return written


def coverage_report(values):
    """Which numbers are real and which are class defaults — print it with the asset."""
    lines = []
    for name, v in values.items():
        prov = v.get("provenance", {})
        rec = [k for k, s in prov.items() if s.startswith("recovered")]
        dfl = [k for k, s in prov.items() if not s.startswith("recovered")]
        lines.append(f"  {name:14s} recovered: {', '.join(rec) if rec else '-':32s} "
                     f"default: {', '.join(dfl) if dfl else '-'}")
    return "\n".join(lines)
