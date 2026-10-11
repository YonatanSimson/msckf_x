#!/usr/bin/env bash
# Builds, runs the unit tests, then every gate checker. Each gate writes
# gate_out/<gate>/report.md; the script exits non-zero if anything fails.
#
#   scripts/run_gates.sh [GATE...]
#
# GATE: stage0_reprojection, stage1a_static_init, stage1_tracker, stage1b_dynamic_init, stage3_sim (default: all;
# stage1_tracker reads stage1a_static_init's output, so keep that order). Data comes
# from scripts/data/fetch_tumvi.sh and scripts/data/fetch_euroc.sh. Python is
# $MSCKF_PYTHON (default ~/.venv/msckf/bin/python, with python/ installed:
# pip install -e python/).
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
build="${MSCKF_BUILD:-$repo/build}"
python="${MSCKF_PYTHON:-$HOME/.venv/msckf/bin/python}"
gates=("$@")
[ ${#gates[@]} -eq 0 ] && gates=(stage0_reprojection stage1a_static_init stage1_tracker stage1b_dynamic_init stage3_sim)

# The GTSAM cross-check test is built when scripts/deps/build_gtsam.sh has run.
gtsam_dir="${GTSAM_DIR:-$repo/build-gtsam/install/lib/cmake/GTSAM}"
cmake_args=()
[ -d "$gtsam_dir" ] && cmake_args+=("-DGTSAM_DIR=$gtsam_dir")

status=0
cmake -S "$repo" -B "$build" -G Ninja "${cmake_args[@]}" >/dev/null && cmake --build "$build" || exit 1
(cd "$build" && ctest --output-on-failure) || status=1

for g in "${gates[@]}"; do
  echo
  echo "=== $g"
  "$python" -m "msckf_gates.$g" || status=1
done

echo
for g in "${gates[@]}"; do
  r="${MSCKF_GATE_OUT:-$repo/gate_out}/$g/report.md"
  [ -f "$r" ] && printf '%-24s %s\n' "$g" "$(grep -m1 -oE 'PASS|FAIL' "$r")"
done
exit $status
