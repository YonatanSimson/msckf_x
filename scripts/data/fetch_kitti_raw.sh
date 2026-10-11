#!/usr/bin/env bash
# Downloads KITTI raw drives in their unsynced "extract" form (raw, distorted
# images and the OXTS GPS/IMU at 100 Hz; the synced form has the IMU at only
# 10 Hz) plus the day's calibration, and extracts them. KITTI is CC BY-NC-SA
# 3.0: evaluation only, never shipped.
#
#   scripts/data/fetch_kitti_raw.sh [-d DEST] [DRIVE...]
#
# DRIVE: e.g. 2011_09_26_drive_0005. Default: city and residential drives with
# turns: 2011_09_26_drive_{0005,0009,0019,0035,0039} (about 9.8 GB of zips).
# DEST defaults to $MSCKF_DATA_LARGE/kitti/raw, with MSCKF_DATA_LARGE
# defaulting to /mnt/c/Users/user/data. The archives publish no checksum; the
# zips' own CRCs are checked (unzip -t). Drives already extracted are skipped.
# Convert with python/kitti_raw_to_euroc.py for the EuRoC-format tools.
set -euo pipefail

dest="${MSCKF_DATA_LARGE:-/mnt/c/Users/user/data}/kitti/raw"
while getopts "d:h" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
drives=("$@")
[ ${#drives[@]} -eq 0 ] && drives=(2011_09_26_drive_0005 2011_09_26_drive_0009 2011_09_26_drive_0019
                                   2011_09_26_drive_0035 2011_09_26_drive_0039)
base=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
mkdir -p "$dest"

fetch() {  # url file
  if [ ! -f "$2" ]; then
    echo "fetching $(basename "$2")"
    curl -fSL --progress-bar -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
  unzip -tq "$2" > /dev/null || { echo "$2: CRC check failed"; exit 1; }
}

for drive in "${drives[@]}"; do
  day="${drive:0:10}"
  if [ ! -f "$dest/$day/calib_cam_to_cam.txt" ]; then
    fetch "$base/${day}_calib.zip" "$dest/${day}_calib.zip"
    unzip -q -o "$dest/${day}_calib.zip" -d "$dest"
  fi
  if [ -d "$dest/$day/${drive}_extract/oxts" ]; then
    echo "$drive: already present"
    continue
  fi
  fetch "$base/$drive/${drive}_extract.zip" "$dest/${drive}_extract.zip"
  unzip -q -o "$dest/${drive}_extract.zip" -d "$dest"
  rm "$dest/${drive}_extract.zip"
  echo "$drive: extracted to $dest/$day/${drive}_extract"
done
