#!/usr/bin/env bash
# Downloads TUM-VI sequences (EuRoC export, 512x512, 16-bit), verifies their
# MD5 and extracts them, plus the published Kalibr calibrations and the
# AprilGrid target description.
#
#   scripts/data/fetch_tumvi.sh [-d DEST] [SEQUENCE...]
#
# SEQUENCE is a TUM-VI name without the "dataset-" prefix and resolution
# suffix, e.g. room1, calib-cam1, corridor1. Default: calib-cam1 room1.
# DEST defaults to $MSCKF_DATA/tumvi, with MSCKF_DATA defaulting to ~/data.
# Sequences already extracted are skipped.
set -euo pipefail

dest="${MSCKF_DATA:-$HOME/data}/tumvi"
while getopts "d:h" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    h) sed -n '2,12p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
sequences=("$@")
[ ${#sequences[@]} -eq 0 ] && sequences=(calib-cam1 room1)

seq_base=https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16
calib_base=https://cvg.cit.tum.de/_media/data/datasets/visual-inertial-dataset

mkdir -p "$dest"

# Published calibrations (Kalibr camchain/imu/results) and the target.
mkdir -p "$dest/kalibr_results"
for f in pinhole-equi-512.zip omni-radtan-512.zip april_6x6_80x80cm.zip; do
  if [ ! -f "$dest/kalibr_results/$f" ]; then
    echo "fetching $f"
    curl -fsSL -o "$dest/kalibr_results/$f" "$calib_base/$f"
  fi
  unzip -q -o "$dest/kalibr_results/$f" -d "$dest/kalibr_results"
done

for s in "${sequences[@]}"; do
  name="dataset-${s}_512_16"
  if [ -d "$dest/$name/mav0" ]; then
    echo "$name: already present"
    continue
  fi
  tmp=$(mktemp -d "$dest/.fetch-$s.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  echo "$name: downloading"
  curl -fsSL -o "$tmp/$name.tar.md5" "$seq_base/$name.tar.md5"
  curl -fSL --progress-bar -o "$tmp/$name.tar" "$seq_base/$name.tar"
  (cd "$tmp" && md5sum -c "$name.tar.md5")
  tar xf "$tmp/$name.tar" -C "$dest"
  rm -rf "$tmp"
  trap - EXIT
  echo "$name: extracted to $dest/$name"
done
