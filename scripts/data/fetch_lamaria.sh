#!/usr/bin/env bash
# Downloads LaMAria (ETH/Meta, Project Aria glasses) training sequences in the
# ASL/EuRoC export (undistorted pinhole stereo + IMU), together with their
# pinhole calibration, pseudo-dense ground truth and control points.
#
#   scripts/data/fetch_lamaria.sh [-d DEST] [SEQUENCE...]
#
# SEQUENCE is a training sequence name, e.g. R_01_easy, R_08_hard,
# sequence_1_19. Default: all training sequences (~170 GB).
# DEST defaults to $MSCKF_DATA/lamaria, with MSCKF_DATA defaulting to ~/data.
# Layout: DEST/training/SEQ/{aria/{cam0,cam1,imu0},pinhole_calibrations,
# ground_truth/{pGT,control_points}}. Sequences already extracted are skipped.
set -euo pipefail

dest="${MSCKF_DATA:-$HOME/data}/lamaria"
while getopts "d:h" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    h) sed -n '2,12p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))

base=https://cvg-data.inf.ethz.ch/lamaria

list_training() {
  curl -fsSL "$base/asl_folder/training/" \
    | grep -o 'href="[^"?/]*\.zip"' | sed 's/href="\(.*\)\.zip"/\1/'
}

# Fetches $1 to $2 unless absent on the server (GT exists only for some).
fetch_optional() {
  [ -f "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fsL -o "$2" "$1" && return 0
  rm -f "$2"
  rmdir -p --ignore-fail-on-non-empty "$(dirname "$2")"
  echo "  (not on server: ${1#"$base"/})"
}

remote_size() {
  curl -fsSIL "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}'
}

fetch_sequence() {
  local s=$1 dir="$dest/training/$1"
  fetch_optional "$base/pinhole_calibrations/training/$s.json" "$dir/pinhole_calibrations/$s.json"
  fetch_optional "$base/ground_truth/pseudo_dense/$s.txt" "$dir/ground_truth/pGT/$s.txt"
  fetch_optional "$base/ground_truth/sparse/$s.json" "$dir/ground_truth/control_points/$s.json"
  if [ -f "$dir/aria/imu0/data.csv" ]; then
    echo "$s: already present"
    return
  fi
  local url="$base/asl_folder/training/$s.zip" zip="$dest/.download/$s.zip"
  mkdir -p "$dest/.download"
  echo "$s: downloading"
  curl -fSL --progress-bar -C - -o "$zip" "$url"
  local want have
  want=$(remote_size "$url")
  have=$(stat -c %s "$zip")
  [ "$want" = "$have" ] || { echo "$s: size mismatch ($have != $want)" >&2; exit 1; }
  unzip -q -o "$zip" -d "$dest/training"
  rm -f "$zip"
  echo "$s: extracted to $dir"
}

sequences=("$@")
[ ${#sequences[@]} -eq 0 ] && mapfile -t sequences < <(list_training)
[ ${#sequences[@]} -gt 0 ] || { echo "no sequences found at $base" >&2; exit 1; }

for s in "${sequences[@]}"; do
  fetch_sequence "$s"
done
rmdir "$dest/.download" 2>/dev/null || true
