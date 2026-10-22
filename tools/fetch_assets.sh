#!/bin/bash
# Download the scene and object models and convert them to USD:
#   * Baylands park (Gazebo Fuel, OpenRobotics, CC-BY 4.0), ~150 MB, with ground colliders
#   * cars: Hatchback red/blue, SUV, Prius Hybrid (Gazebo Fuel, OpenRobotics)
#   * people: Standing person, Casual female, Walking person, Rescue Randy (Gazebo Fuel, OpenRobotics)
#   * people: 7 Isaac Sim 6.0 characters (NVIDIA, see tools/fetch_isaac_people.sh)
# Needs curl, unzip, python3 (creates tools/.venv with usd-core, trimesh, pycollada).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/assets/src"; mkdir -p "$SRC"
PY="$HERE/tools/.venv/bin/python"
if [ ! -x "$PY" ]; then
    python3 -m venv "$HERE/tools/.venv"
    "$HERE/tools/.venv/bin/pip" install -q -r "$HERE/tools/requirements.txt"
fi
fetch() {  # fetch <Fuel model name> -> assets/src/<lower_snake_name>
    local d; d=$(echo "$1" | tr 'A-Z ' 'a-z_')
    [ -d "$SRC/$d" ] && return
    curl -sL -o "$SRC/$d.zip" "https://fuel.gazebosim.org/1.0/OpenRobotics/models/${1// /%20}/tip/${1// /%20}.zip"
    mkdir -p "$SRC/$d" && unzip -o -q "$SRC/$d.zip" -d "$SRC/$d" && rm "$SRC/$d.zip"
}
conv() {  # conv <src dir> <mesh file> <asset name> [mesh_to_usd args]: meshes + textures side by side, then convert
    local s="$SRC/$1" m="$2" o="$3"; shift 3
    local T; T=$(mktemp -d)
    cp "$s"/meshes/* "$T/"; cp "$s"/materials/textures/* "$T/" 2>/dev/null || true
    cp -n "$SRC/hatchback/materials/textures/wheels3.png" "$T/" 2>/dev/null || true
    sed -i -E 's#(\.\./materials/textures/|https://[^ ]*/textures/)##g' "$T"/*.mtl "$T"/*.dae 2>/dev/null || true
    "$PY" "$HERE/tools/mesh_to_usd.py" "$T/$m" "$HERE/assets/$o/$o.usd" --center --ground "$@"
    rm -rf "$T"
}
echo "Downloading from Gazebo Fuel..."
for m in "Baylands" "Hatchback" "Hatchback red" "Hatchback blue" "SUV" "Prius Hybrid" \
         "Standing person" "Casual female" "Walking person" "Rescue Randy"; do fetch "$m"; done

conv hatchback_red hatchback.obj car_hatchback_red --scale 0.0254 --rotz 90
conv hatchback_blue hatchback.obj car_hatchback_blue --scale 0.0254 --rotz 90
conv suv suv.obj car_suv --scale 0.06 --rotz 90
conv prius_hybrid Hybrid.obj car_prius --scale 0.01 --rotz 90
conv standing_person standing.dae person_standing
conv casual_female casual_female.dae person_casual_female
conv walking_person walking.dae person_walking
conv rescue_randy rescue_randy.dae person_rescue_randy --scale 0.01

if [ ! -e "$HERE/assets/baylands/baylands.usd" ]; then
    # three DAE tiles, lowered 1.3 m as in the Fuel SDF, then triangle-mesh colliders on the ground meshes
    "$PY" "$HERE/tools/mesh_to_usd.py" "$SRC"/baylands/media/baylands_0{1,2,3}.DAE "$HERE/assets/baylands/baylands.usd" --z-offset -1.3
    "$PY" "$HERE/tools/add_colliders.py" "$HERE/assets/baylands/baylands.usd" "$SRC/baylands/media/Textures"
fi
"$HERE/tools/fetch_isaac_people.sh"
echo "Assets ready in $HERE/assets"
