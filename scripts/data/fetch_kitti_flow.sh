#!/usr/bin/env bash
# Downloads the KITTI 2015 scene flow / optical flow benchmark (CC BY-NC-SA
# 3.0: evaluation only, never shipped) and extracts it. The 200 training
# pairs (frame 10 -> 11) carry LiDAR-based ground-truth flow, used for the
# tracker's single-step accuracy on real driving images.
#
#   scripts/data/fetch_kitti_flow.sh [-d DEST]
#
# DEST defaults to $MSCKF_DATA_LARGE/kitti/flow2015, with MSCKF_DATA_LARGE
# defaulting to /mnt/c/Users/user/data. The archive publishes no checksum; the
# zip's own CRCs are checked.
set -euo pipefail

dest="${MSCKF_DATA_LARGE:-/mnt/c/Users/user/data}/kitti/flow2015"
while getopts "d:h" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    h) sed -n '2,12p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

base=https://s3.eu-central-1.amazonaws.com/avg-kitti
mkdir -p "$dest"
cd "$dest"
# Images + ground-truth flow, and the per-pair camera calibration (intrinsics,
# needed for bearings).
for name in data_scene_flow data_scene_flow_calib; do
  if [ -f ".complete-$name" ]; then
    echo "$name: already extracted in $dest"
    continue
  fi
  if [ ! -f "$name.zip" ]; then
    curl -fSL --progress-bar -o "$name.zip.part" "$base/$name.zip"
    mv "$name.zip.part" "$name.zip"
  fi
  unzip -tq "$name.zip"
  ok=0
  for attempt in 1 2 3; do
    if unzip -q -o "$name.zip" -d . </dev/null; then ok=1; break; fi
    echo "$name: unzip failed (attempt $attempt)"
  done
  [ "$ok" -eq 1 ] || { echo "$name: extraction failed" >&2; exit 1; }
  touch ".complete-$name"
  echo "$name: extracted to $dest"
done
