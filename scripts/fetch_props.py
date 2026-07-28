"""Fetch a mixed-physics prop set from Poly Haven (CC0) for the scene milestone.

The set is chosen so the objects LOOK like ordinary scene dressing but must move very
differently — which is the whole point of recovering per-object values rather than
guessing one default: a brass pot is heavy and dead, a cardboard box is light, a
rubber duck is squishy, a baseball bounces, a pillow drapes.

Run: python scripts/fetch_props.py            (any env with stdlib only)
"""
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (simulation-assetization research asset fetch)"}
API = "https://api.polyhaven.com/files/{}"

# name -> (folder, what it is for the physics story)
PROPS = {
    "rubber_duck_toy":           ("soft",  "the duck — squishy, the pitch's own example"),
    "throw_pillows_01":          ("soft",  "pillow — drapes/compresses"),
    "baseball_01":               ("rigid", "bouncy, dense, small"),
    "food_apple_01":             ("rigid", "rolls, light"),
    "ceramic_vase_01":           ("rigid", "rigid + dense, tips rather than slides"),
    "brass_pot_01":              ("rigid", "heavy metal — looks like the vase, moves nothing like it"),
    "cardboard_box_01":          ("rigid", "light, slides, high friction"),
    "book_encyclopedia_set_01":  ("rigid", "flat, slides, never bounces"),
    "wooden_bowl_01":            ("rigid", "wood — mid density"),
    "coffee_cart":               ("rigid", "larger prop (skipped if missing)"),
}
RES = "2k"


def get(url, timeout=180):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def fetch_model(name, folder):
    dst = REPO / "assets" / folder / name
    if (dst / f"{name}_{RES}.gltf").exists():
        print(f"  skip {name} (already present)")
        return True
    try:
        files = json.load(get(API.format(name), timeout=60))
    except Exception as e:
        print(f"  MISS {name}: {e}")
        return False
    node = files.get("gltf", {}).get(RES, {}).get("gltf")
    if not node:
        print(f"  MISS {name}: no gltf/{RES}")
        return False
    (dst / "textures").mkdir(parents=True, exist_ok=True)
    urls = [node["url"]] + [inc["url"] for inc in (node.get("include") or {}).values()]
    for u in urls:
        fn = u.split("/")[-1]
        out = dst / ("textures/" + fn if (".jpg" in fn or ".png" in fn) else fn)
        try:
            data = get(u).read()
        except Exception as e:
            print(f"  warn {name}/{fn}: {e}")
            continue
        out.write_bytes(data)
    n = len(list(dst.rglob("*")))
    print(f"  OK   {name:26s} -> assets/{folder}/{name}  ({n} files)")
    return True


def main():
    print(f"fetching {len(PROPS)} Poly Haven props (CC0)")
    ok = 0
    for name, (folder, why) in PROPS.items():
        print(f"- {name}: {why}")
        ok += bool(fetch_model(name, folder))
    print(f"\n{ok}/{len(PROPS)} fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
