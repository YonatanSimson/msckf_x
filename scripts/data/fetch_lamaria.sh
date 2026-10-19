#!/usr/bin/env bash
# Downloads LaMAria (ETH/Meta, Project Aria glasses) training sequences in the
# ASL/EuRoC export (undistorted pinhole stereo + IMU), together with their
# pinhole calibration, pseudo-dense ground truth and control points.
# With -r, also the original Aria recordings (fisheye VRS, ~124 GB in all) and
# their Aria calibration (camera model FISHEYE624, RAD_TAN_THIN_PRISM_FISHEYE).
#
#   scripts/data/fetch_lamaria.sh [-d DEST] [-r] [SEQUENCE...]
#
# SEQUENCE is a training sequence name, e.g. R_01_easy, R_08_hard,
# sequence_1_19. Default: all training sequences (~170 GB).
# DEST defaults to $MSCKF_DATA_LARGE/lamaria, with MSCKF_DATA_LARGE defaulting
# to /mnt/c/Users/user/data.
# Layout: DEST/training/SEQ/{aria/{cam0,cam1,imu0},pinhole_calibrations,
# ground_truth/{pGT,control_points},aria_calibrations,raw/SEQ.vrs}.
# Sequences already extracted, and recordings already complete, are skipped.
set -euo pipefail

dest="${MSCKF_DATA_LARGE:-/mnt/c/Users/user/data}/lamaria"
raw=0
while getopts "d:rh" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    r) raw=1 ;;
    h) sed -n '2,16p' "$0"; exit 0 ;;
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
# Downloads to $2.part first, so an interrupted fetch is retried next time;
# any failure other than a 404 stops the script.
fetch_optional() {
  [ -f "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  local code
  code=$(curl -sL -o "$2.part" -w '%{http_code}' "$1") || code=error
  if [ "$code" = 200 ]; then
    mv "$2.part" "$2"
    return 0
  fi
  rm -f "$2.part"
  rmdir -p --ignore-fail-on-non-empty "$(dirname "$2")"
  if [ "$code" = 404 ]; then
    echo "  (not on server: ${1#"$base"/})"
    return 0
  fi
  echo "${1#"$base"/}: download failed (HTTP $code)" >&2
  exit 1
}

# Extraction finished: the marker written after unzip, or (sequences extracted
# before the marker existed) every sensor's data.csv present.
extracted() {
  [ -f "$1/.extracted" ] && return 0
  [ -f "$1/aria/imu0/data.csv" ] && [ -f "$1/aria/cam0/data.csv" ] && [ -f "$1/aria/cam1/data.csv" ]
}

remote_size() {
  curl -fsSIL "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}'
}

# Downloads $1 to $2, resuming a partial download, and checks its size
# against the server's.
download() {
  echo "${2##*/}: downloading"
  curl -fSL --progress-bar -C - -o "$2" "$1"
  local want have
  want=$(remote_size "$1")
  have=$(stat -c %s "$2")
  if [ -z "$want" ]; then
    echo "${2##*/}: server sent no size, size check skipped" >&2
  elif [ "$want" != "$have" ]; then
    echo "${2##*/}: size mismatch ($have != $want)" >&2
    exit 1
  fi
}

# The original Aria recording, kept as downloaded (.part until complete).
fetch_recording() {
  local s=$1 vrs="$dest/training/$1/raw/$1.vrs"
  fetch_optional "$base/aria_calibrations/training/$s.json" "$dest/training/$s/aria_calibrations/$s.json"
  if [ -f "$vrs" ]; then
    echo "$s.vrs: already present"
    return
  fi
  mkdir -p "${vrs%/*}"
  download "$base/raw_data/training/$s.vrs" "$vrs.part"
  mv "$vrs.part" "$vrs"
}

fetch_sequence() {
  local s=$1 dir="$dest/training/$1"
  fetch_optional "$base/pinhole_calibrations/training/$s.json" "$dir/pinhole_calibrations/$s.json"
  fetch_optional "$base/ground_truth/pseudo_dense/$s.txt" "$dir/ground_truth/pGT/$s.txt"
  fetch_optional "$base/ground_truth/sparse/$s.json" "$dir/ground_truth/control_points/$s.json"
  if extracted "$dir"; then
    echo "$s: already present"
    return
  fi
  local zip="$dest/.download/$s.zip"
  mkdir -p "$dest/.download"
  download "$base/asl_folder/training/$s.zip" "$zip"
  unzip -q -o "$zip" -d "$dest/training"
  touch "$dir/.extracted"
  rm -f "$zip"
  echo "$s: extracted to $dir"
}

sequences=("$@")
[ ${#sequences[@]} -eq 0 ] && mapfile -t sequences < <(list_training)
[ ${#sequences[@]} -gt 0 ] || { echo "no sequences found at $base" >&2; exit 1; }

for s in "${sequences[@]}"; do
  fetch_sequence "$s"
  if [ "$raw" = 1 ]; then fetch_recording "$s"; fi
done
rmdir "$dest/.download" 2>/dev/null || true
