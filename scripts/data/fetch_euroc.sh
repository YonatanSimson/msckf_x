#!/usr/bin/env bash
# Downloads EuRoC MAV sequences from the ETH Research Collection and extracts
# them in ASL format (mav0/...). EuRoC is distributed as bundles (one zip per
# environment, holding a ROS bag and an ASL zip per sequence); only the ASL
# zips are kept.
#
#   scripts/data/fetch_euroc.sh [-d DEST] [-i] [BUNDLE...]
#
# BUNDLE: machine_hall (12.7 GB), vicon_room1 (6.0 GB), vicon_room2 (6.0 GB).
# Default: machine_hall. DEST defaults to $MSCKF_DATA_LARGE/euroc, with
# MSCKF_DATA_LARGE defaulting to /mnt/c/Users/user/data (the bundles are too
# large for the WSL disk budget). -i extracts only IMU and ground truth (fast;
# enough for static initialization).
set -euo pipefail

dest="${MSCKF_DATA_LARGE:-/mnt/c/Users/user/data}/euroc"
imu_only=0
while getopts "d:ih" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    i) imu_only=1 ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
bundles=("$@")
[ ${#bundles[@]} -eq 0 ] && bundles=(machine_hall)

# Research Collection record 20.500.11850/690084 (DSpace bitstream ids).
declare -A url=(
  [machine_hall]=https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content
  [vicon_room1]=https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content
  [vicon_room2]=https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content
)

mkdir -p "$dest"
cd "$dest"
for b in "${bundles[@]}"; do
  [ -n "${url[$b]:-}" ] || { echo "unknown bundle $b" >&2; exit 2; }
  if [ ! -f "$b.zip" ]; then
    echo "$b: downloading"
    curl -fSL --progress-bar -A "Mozilla/5.0" -o "$b.zip.part" "${url[$b]}"
    mv "$b.zip.part" "$b.zip"
  fi
  if [ ! -f "$b.zip.verified" ]; then
    unzip -tq "$b.zip"  # integrity (CRC) check; the record publishes no checksums
    touch "$b.zip.verified"
  fi
  # Inner ASL zips, e.g. machine_hall/MH_01_easy/MH_01_easy.zip -> ./MH_01_easy.zip
  unzip -q -o -j "$b.zip" "$b/*/*.zip" -d .
  for z in $(unzip -Z1 "$b.zip" "$b/*/*.zip" | xargs -n1 basename); do
    s="${z%.zip}"
    if [ "$imu_only" -eq 1 ]; then
      members=('mav0/imu0/*' 'mav0/state_groundtruth_estimate0/*')
      marker="$s/.complete-imu"
    else
      members=('*')
      marker="$s/.complete"
    fi
    if [ -f "$marker" ] || [ -f "$s/.complete" ]; then
      echo "$s: already extracted"
      continue
    fi
    mkdir -p "$s"
    # Writes to /mnt/c (9P) occasionally fail transiently ("write error (disk
    # full?)" with plenty of space). unzip then exits non-zero; retry the
    # whole sequence (overwriting) and only mark it complete after a clean run.
    ok=0
    for attempt in 1 2 3; do
      echo "$s: extracting (attempt $attempt)"
      if unzip -q -o "$z" "${members[@]}" -d "$s" </dev/null; then
        ok=1
        break
      fi
    done
    if [ "$ok" -ne 1 ]; then
      echo "$s: extraction failed after 3 attempts" >&2
      exit 1
    fi
    touch "$marker"
    echo "$s: extracted to $dest/$s"
  done
done
