"""Real-asset loading: GSO / PolyHaven meshes -> numpy geometry -> Newton scenes.

Asset inventory layout (see assets/):
  assets/rigid/<Name>/meshes/model.obj   (GSO: real scanned objects, obj + texture)
  assets/cloth/<Name>/...                (GSO scans of cloth-like items: towel, cushion)
  assets/scenes/<name>/                  (PolyHaven CC0: table gltf, HDRI)

GSO scans are in meters, Z-up, watertight-ish; trimesh handles obj/gltf/glb.
"""

from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"


def list_assets():
    """{'rigid': [names], 'cloth': [...], 'scenes': [...]}"""
    out = {}
    for cat in ("rigid", "cloth", "scenes"):
        d = ASSETS / cat
        out[cat] = sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []
    return out


def _find_mesh_file(asset_dir: Path) -> Path:
    for pattern in ("meshes/model.obj", "*.obj", "*.gltf", "*.glb"):
        hits = sorted(asset_dir.glob(pattern)) or sorted(asset_dir.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no mesh found under {asset_dir}")


# glTF is Y-up by specification. Blender's importer converts to Z-up on load; trimesh
# does not, so for the whole project the simulator held meshes in a different orientation
# from the renderer. The ceramic vase was simulated 12.6 x 26.5 cm on the ground and
# 13.1 cm tall -- lying on its side -- while every rendered frame showed it upright, and
# the wooden bowl stood on its rim. Sphere covers, resting heights and all contact
# geometry were built from the wrong pose.
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])


def load_asset(category: str, name: str, up_convert: bool = True) -> trimesh.Trimesh:
    """Load an asset as a single concatenated trimesh (geometry only).

    up_convert applies the glTF Y-up -> Z-up rotation so the mesh matches the orientation
    Blender renders it in. Pass False only to inspect the file as authored.
    """
    mesh_path = _find_mesh_file(ASSETS / category / name)
    m = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):  # pragma: no cover - force="mesh" should prevent
        m = m.to_mesh()
    if up_convert and str(mesh_path).lower().endswith((".gltf", ".glb")):
        m.apply_transform(_YUP_TO_ZUP)
    return m


def asset_stats(m: trimesh.Trimesh) -> dict:
    ext = m.bounds[1] - m.bounds[0]
    return {
        "vertices": len(m.vertices),
        "faces": len(m.faces),
        "extent_m": np.round(ext, 4).tolist(),
        "max_dim_m": float(ext.max()),
        "watertight": bool(m.is_watertight),
        "volume_m3": float(m.volume) if m.is_watertight else None,
    }


def decimate(m: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Simplify for simulation (GSO scans are render-resolution)."""
    if len(m.faces) <= target_faces:
        return m
    return m.simplify_quadric_decimation(face_count=target_faces)


def add_rigid_asset(builder, m: trimesh.Trimesh, pos, rot=None, density: float = 500.0,
                    target_faces: int = 2000):
    """Add a (decimated) asset as a rigid body + collision mesh to a Newton builder."""
    import warp as wp

    import newton

    sim_mesh = decimate(m, target_faces)
    mesh = newton.Mesh(sim_mesh.vertices.astype(np.float32), sim_mesh.faces.ravel().astype(np.int32))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*pos), rot if rot is not None else wp.quat_identity())
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=density)
    builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
    return body
