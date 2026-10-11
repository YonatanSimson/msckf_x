#!/usr/bin/env bash
# Downloads TartanAir v2 (CC BY 4.0) left-front camera images and depth for a
# set of environments, verifies each zip against its storage ETag (MD5) and
# extracts it. Exact tracking ground truth is computed from depth and poses
# (tools/tracker_eval); TartanAir no longer ships precomputed flow.
#
#   scripts/data/fetch_tartanair.sh [-d DEST] [ENV/DIFFICULTY...]
#
# ENV/DIFFICULTY like RetroOffice/hard. Default: the tracker benchmark set
# below. DEST defaults to $MSCKF_DATA_LARGE/tartanair, with MSCKF_DATA_LARGE
# defaulting to /mnt/c/Users/user/data.
#
# Storage: the AirLab Swift bucket used by the official toolkit
# (tartanair 1.4.0, tartanair/downloader.py), read anonymously over HTTPS.
set -euo pipefail

dest="${MSCKF_DATA_LARGE:-/mnt/c/Users/user/data}/tartanair"
while getopts "d:h" opt; do
  case "$opt" in
    d) dest="$OPTARG" ;;
    h) sed -n '2,15p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
sets=("$@")
[ ${#sets[@]} -eq 0 ] && sets=(RetroOffice/easy RetroOffice/hard Office/hard OldTownFall/easy
                               OldTownFall/hard OldTownNight/hard AmericanDiner/hard)

base=https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair_v2
mkdir -p "$dest"
cd "$dest"
for s in "${sets[@]}"; do
  env="${s%/*}"
  diff="${s#*/}"
  for f in image_lcam_front depth_lcam_front; do
    key="$env/Data_$diff/$f.zip"
    zip="$dest/$key"
    marker="$dest/$env/Data_$diff/.complete-$f"
    if [ -f "$marker" ]; then
      echo "$key: already extracted"
      continue
    fi
    mkdir -p "$(dirname "$zip")"
    etag=$(curl -fsSI "$base/$key" | tr -d '\r' | awk 'tolower($1)=="etag:" {gsub(/"/,"",$2); print $2}')
    if [ ! -f "$zip" ] || [ "$(md5sum "$zip" | cut -d' ' -f1)" != "$etag" ]; then
      echo "$key: downloading"
      curl -fSL --progress-bar -o "$zip.part" "$base/$key"
      mv "$zip.part" "$zip"
    fi
    [ "$(md5sum "$zip" | cut -d' ' -f1)" = "$etag" ] || { echo "$key: MD5 mismatch" >&2; exit 1; }
    # The zips hold <env>/Data_<diff>/P00x/... paths; retry transient /mnt/c
    # write errors as in fetch_euroc.sh.
    ok=0
    for attempt in 1 2 3; do
      if unzip -q -o "$zip" -d "$dest" </dev/null; then ok=1; break; fi
      echo "$key: unzip failed (attempt $attempt)"
    done
    [ "$ok" -eq 1 ] || { echo "$key: extraction failed" >&2; exit 1; }
    touch "$marker"
    echo "$key: extracted"
  done
done
