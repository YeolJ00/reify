# Mesh contact replaces the sphere cover

## The problem

Contact geometry was a voxelisation of each mesh into interior spheres of one common
radius; the mesh was then never touched again. Measured against sampled surface points:

| object | spheres | mean error | max error |
|---|---|---|---|
| brass pot | 1068 | 1.99 mm | 7.50 mm |
| wooden bowl | 590 | 2.01 mm | 7.27 mm |
| baseball | 74 | 2.48 mm | 9.55 mm |
| C-clamp | 107 | 2.83 mm | 10.61 mm |
| cardboard box | 2435 | 2.09 mm | 7.09 mm |

**2.28 mm mean.** Three separate problems this session traced back to it: objects rendered
floating (the cover sits below the visible surface), "contact area" was a voxel-resolution
artifact from 1 to 458 spheres that the stiffness calibration keyed off, and a flat
object's contact surface was bumpy so slip angle carried a geometric term that is not
friction.

If the geometry is approximated by spheres there is little reason to simulate authored
meshes at all — the mesh *is* the asset.

## The replacement

`src/sim/mesh_contact.py`, `src/sim/mesh_scene.py`.

**Ground contact is exact.** The table is a plane and a linear function over a triangle
attains its minimum at a vertex, so testing every mesh vertex against the plane is exact —
no queries, no radius, no approximation.

**Body-body uses Warp's mesh BVH.** `wp.mesh_query_point_sign_normal` +
`wp.mesh_eval_position` give signed distance and closest point on the real triangles. The
query runs in the *other* body's local frame — the world vertex is pulled back through that
body's transform — so each `wp.Mesh` is built once at setup and never rebuilt or refitted
as bodies move. That is what makes it affordable per step.

**Forces are area-weighted.** Each vertex carries its Voronoi share of surface area,
normalised per body. Without it a densely-tesselated region pushes harder than a sparse one
for no physical reason — the same defect as counting spheres. It also removes the
contact-count factor from stiffness calibration entirely.

## Result

Slip angle vs `atan(μ)`, wooden bowl, five friction values:

| | raw mean error | after removing the detector offset |
|---|---|---|
| sphere cover | 5.31° (offset 3.97°) | 1.34° |
| **mesh contact** | 3.15° (offset 3.14°) | **0.79°** |

The residual offset is the onset detector firing only after the body has moved 5% of its
size — a detector property, not a physics error.

## Three probes, and one of them does not work

`scripts/mesh_probes.py` runs tilt, drop and collide on the same 14-object scene.

**tilt — works.** Slip angle recovers `atan(μ_eff)`; errors −5.1° to +2.4° across the seven
sliding objects.

**drop — weak.** `ρ(log cd, rebound) = −0.186`. The sign is right but the magnitude is not
usable: rebound height is dominated by shape (a bowl and a book land differently whatever
their damping).

**collide — measures nothing, by construction.** `ρ(mass, travel) = +0.181`. This is not a
tuning problem. A body shoved at fixed *velocity* decelerates at `μg`, which is
mass-independent, so it slides `d = v²/(2μg)` **regardless of its mass**. The probe cannot
read mass no matter how it is weighted.

To make mass observable the excitation must be a fixed *impulse* rather than a fixed
velocity, or the reading must come from momentum transfer between two bodies — object A
struck by a known projectile, where the velocity split depends on the mass ratio. That is
the probe worth building next; it is the only route to mass in this rig.
