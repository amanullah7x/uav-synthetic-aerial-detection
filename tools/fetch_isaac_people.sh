#!/bin/bash
# Download a few Isaac Sim 6.0 character USDs (with their textures) from NVIDIA's public asset bucket with curl.
# Kit's own S3 download is slow/unreliable on some connections, so the files are mirrored locally once.
#   tools/fetch_isaac_people.sh [dest]      (default: assets/isaac_people)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HERE/assets/isaac_people}"
R=https://omniverse-content-production.s3-us-west-2.amazonaws.com
P=Assets/Isaac/6.0/Isaac/People/Characters
CHARS="F_Business_02 F_Medical_01 M_Medical_01 female_adult_police_02 male_adult_construction_03 male_adult_police_04 original_male_adult_construction_05"
for c in $CHARS; do
    [ -d "$DEST/$c" ] && continue
    keys=$(curl -s -m 60 "$R/?list-type=2&prefix=$P/$c/" | grep -oE "<Key>[^<]+" | sed 's/<Key>//' | grep -v "/\.thumbs/")
    for k in $keys; do
        out="$DEST/${k#$P/}"; mkdir -p "$(dirname "$out")"
        curl -s --retry 3 -o "$out" "$R/$k"
    done
    echo "$c: $(echo "$keys" | wc -l) files"
done
