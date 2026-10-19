#!/usr/bin/env bash
# Runs a baseline system as an unmodified binary (clean-room rule: never read
# its source), monocular, on a ROS1 bag, and writes its estimated IMU
# trajectory in TUM format for tools/msckf_eval.
#
#   scripts/baselines/run_baseline.sh openvins|sqrtvins CONFIG BAG OUT_DIR [BAG_START_S]
#
# CONFIG is a config folder name of the system (euroc_mav, tum_vi,
# uzhfpv_indoor, uzhfpv_indoor_45, uzhfpv_outdoor, uzhfpv_outdoor_45) or the
# path of a folder holding estimator_config.yaml and its kalibr chains
# (python/msckf_gates/stage3_baselines.py writes one per LaMAria sequence).
# BAG comes from python/euroc_to_rosbag.py with that config's topics. OUT_DIR
# gets est.txt (ov_eval pose_to_file: "# t tx ty tz qx qy qz qw") and run.log.
# Needs the user-space ROS1 (RoboStack) environment and the catkin workspaces
# under $MSCKF_NON_COMMERCIAL (default ~/work/non_commercial): see the plan,
# Stage 3 "Baselines". Set ROS_MASTER_URI to a free port to run several at
# once (roslaunch starts a master there).
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,16p' "$0"; exit 2; }
system=$1; config=$2; bag=$3; out=$4; bag_start=${5:-0}
nc="${MSCKF_NON_COMMERCIAL:-$HOME/work/non_commercial}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
case "$system" in
  openvins) ws="$nc/catkin_ws"; src="$nc/open_vins"; pkg=ov_msckf; extra=() ;;
  sqrtvins) ws="$nc/catkin_ws_sqrt"; src="$nc/sqrtVINS"; pkg=ov_srvins; extra=(dolivetraj:=false dotime:=false) ;;
  *) echo "unknown system $system"; exit 2 ;;
esac
mkdir -p "$out"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
set +u  # the conda activation scripts read unset variables
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
micromamba activate ros_noetic
# shellcheck disable=SC1091
source "$ws/devel/setup.bash"
set -u

config_path="$src/config/$config/estimator_config.yaml"
[ -f "$config_path" ] || config_path="$config/estimator_config.yaml"
[ -f "$config_path" ] || { echo "no config $config for $system"; exit 1; }

roslaunch "$pkg" serial.launch config:="$config" config_path:="$config_path" bag:="$bag" bag_start:="$bag_start" \
  max_cameras:=1 use_stereo:=false dosave:=true path_est:="$out/est.txt" verbosity:=INFO "${extra[@]}" \
  > "$out/run.log" 2>&1 || { echo "$system failed, see $out/run.log"; exit 1; }
echo "$out/est.txt: $(grep -vc '^#' "$out/est.txt") poses"
