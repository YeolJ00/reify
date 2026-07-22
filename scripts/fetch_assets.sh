#!/usr/bin/env bash
# Reproducible asset fetch. Binaries are gitignored; this recreates assets/.
# GSO objects: CC-BY 4.0 (Google Scanned Objects, via Gazebo Fuel).
# PolyHaven: CC0.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets/rigid assets/cloth assets/scenes

gso() { # $1 = model name, $2 = target dir
  local url="https://fuel.gazebosim.org/1.0/GoogleResearch/models/$1.zip"
  if [ -d "$2/$1" ]; then echo "skip $1"; return; fi
  curl -sL --max-time 300 -o "/tmp/$1.zip" "$url"
  mkdir -p "$2/$1" && unzip -qo "/tmp/$1.zip" -d "$2/$1" && rm "/tmp/$1.zip"
  echo "OK  $1"
}

for m in Great_Dinos_Triceratops_Toy Pony_C_Clamp_1440 Poppin_File_Sorter_Blue \
         Schleich_Lion_Action_Figure Threshold_Porcelain_Teapot_White \
         Weisshai_Great_White_Shark; do gso "$m" assets/rigid; done
for m in Provence_Bath_Towel_Royal_Blue REEF_BRAIDED_CUSHION; do gso "$m" assets/cloth; done

# PolyHaven wooden table (gltf 2k) + studio HDRI
T=assets/scenes/wooden_table_02
mkdir -p "$T/textures"
base="https://dl.polyhaven.org/file/ph-assets"
curl -sL -o "$T/wooden_table_02_2k.gltf" "$base/Models/gltf/2k/wooden_table_02/wooden_table_02_2k.gltf"
curl -sL -o "$T/wooden_table_02.bin"     "$base/Models/gltf/4k/wooden_table_02/wooden_table_02.bin"
for t in nor_gl diff arm; do
  curl -sL -o "$T/textures/wooden_table_02_${t}_2k.jpg" \
    "$base/Models/jpg/2k/wooden_table_02/wooden_table_02_${t}_2k.jpg"
done
curl -sL -o assets/scenes/studio_small_03_2k.hdr "$base/HDRIs/hdr/2k/studio_small_03_2k.hdr"
echo "assets ready:"; find assets -maxdepth 2 -type d | sort
