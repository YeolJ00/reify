"""Author a tiny USD holding a DomeLight with the scene's HDRI.

ViewerRTX exposes only 'default' / 'studio' / 'none' lighting presets, so the street HDRI
the clips were staged against cannot be selected directly. But add_background_usd()
references an arbitrary USD into the stage, and a DomeLight carrying the same .hdr gives
the viewer the same environment lighting and backdrop Blender used.

Run: python scripts/make_hdri_dome.py            (warp env)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HDR = REPO / "assets" / "scenes" / "city" / "pretville_street_2k.hdr"
OUT = REPO / "outputs" / "scene" / "hdri_dome.usda"


def main():
    from pxr import Usd, UsdGeom, UsdLux, Gf

    if not HDR.exists():
        print(f"missing {HDR}"); return 1
    stage = Usd.Stage.CreateNew(str(OUT)) if not OUT.exists() else Usd.Stage.Open(str(OUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Environment")
    dome = UsdLux.DomeLight.Define(stage, "/Environment/DomeLight")
    dome.CreateTextureFileAttr().Set(str(HDR))
    dome.CreateIntensityAttr().Set(1.0)
    # Blender maps the HDRI with +Z up and the seam behind the camera; a -90 deg X
    # rotation brings USD's Y-up dome convention into the same orientation.
    UsdGeom.Xformable(dome).AddRotateXOp().Set(-90.0)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    print(f"wrote {OUT}  (dome light -> {HDR.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
