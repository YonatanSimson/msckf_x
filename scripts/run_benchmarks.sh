#!/usr/bin/env bash
# Re-runs every benchmark (unit tests, each gate, the tracker comparison) into
# a fresh, dated directory and compares the scores with the previous run, to
# catch degradations.
#
#   scripts/run_benchmarks.sh [-r ROOT] [-b BASELINE_RUN] [BENCHMARK...]
#
# Writes ROOT/<YYYY-MM-DD_HHMMSS>/:
#   manifest.txt            commit, dirty files, host, command
#   diff.patch              uncommitted changes the run was built from
#   <benchmark>/            report.md, metrics.json, run.log and the outputs
#   compare.md              per metric: REGRESSION / DEGRADED / IMPROVED
#                           against the previous run (python/benchmark_compare.py)
#
# BENCHMARK: unit_tests stage0_reprojection stage1a_static_init stage1_tracker
# stage1b_dynamic_init stage3_sim stage3_observability stage3_baselines
# stage3_real basalt_flow_check tracker_compare (default: all, in this order;
# stage1_tracker reads stage1a_static_init's output, stage3_real reads
# stage1b_dynamic_init's tracks and stage3_baselines' measured ATEs from the
# same run; stage3_baselines runs OpenVINS and sqrtVINS as binaries, about
# real time per sequence). ROOT defaults to
# $MSCKF_BENCHMARK_ROOT or ~/benchmark; AprilGrid detections for
# tracker_compare are cached in ROOT/cache/corners. Data comes from
# scripts/data/fetch_*.sh. Exits non-zero if a benchmark fails or a score
# degraded.
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
build="${MSCKF_BUILD:-$repo/build}"
python="${MSCKF_PYTHON:-$HOME/.venv/msckf/bin/python}"
root="${MSCKF_BENCHMARK_ROOT:-$HOME/benchmark}"
baseline=()
while getopts "r:b:h" opt; do
  case "$opt" in
    r) root="$OPTARG" ;;
    b) baseline=(--baseline "$OPTARG") ;;
    h) sed -n '2,23p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
benchmarks=("$@")
all_benchmarks=(unit_tests stage0_reprojection stage1a_static_init stage1_tracker stage1b_dynamic_init stage3_sim
  stage3_observability stage3_baselines stage3_real basalt_flow_check tracker_compare)
[ ${#benchmarks[@]} -eq 0 ] && benchmarks=("${all_benchmarks[@]}")

run="$root/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$run"
{
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "commit: $(git -C "$repo" rev-parse HEAD)"
  echo "dirty files:"
  git -C "$repo" status --porcelain | sed 's/^/  /'
  echo "command: $0 $*"
  echo "benchmarks: ${benchmarks[*]}"
} > "$run/manifest.txt"
git -C "$repo" diff HEAD > "$run/diff.patch"
echo "benchmark run: $run"

# Same build as run_gates.sh (GTSAM cross-check when scripts/deps/build_gtsam.sh has run).
gtsam_dir="${GTSAM_DIR:-$repo/build-gtsam/install/lib/cmake/GTSAM}"
cmake_args=()
[ -d "$gtsam_dir" ] && cmake_args+=("-DGTSAM_DIR=$gtsam_dir")
{ cmake -S "$repo" -B "$build" -G Ninja "${cmake_args[@]}" && cmake --build "$build"; } > "$run/build.log" 2>&1 \
  || { echo "build failed, see $run/build.log"; exit 1; }

status=0
for b in "${benchmarks[@]}"; do
  out="$run/$b"
  mkdir -p "$out"
  echo
  echo "=== $b"
  case "$b" in
    unit_tests)
      (cd "$build" && ctest --output-on-failure) > "$out/run.log" 2>&1
      ok=$?
      total=$(grep -oE 'out of [0-9]+' "$out/run.log" | grep -oE '[0-9]+' | tail -1)
      failed=$(grep -oE '[0-9]+ tests failed' "$out/run.log" | grep -oE '^[0-9]+' | tail -1)
      "$python" -c "
import sys; sys.path.insert(0, '$repo/python')
from msckf_gates.common import write_metrics
write_metrics('$out', [{'name': 'failed tests', 'value': float(${failed:-0}), 'better': 'lower'},
                       {'name': 'tests', 'value': float(${total:-0}), 'better': None}], passed=($ok == 0))
"
      tail -3 "$out/run.log"
      [ $ok -eq 0 ] || status=1
      ;;
    basalt_flow_check)
      "$python" "$repo/python/basalt_flow_check.py" --out "$out" > "$out/run.log" 2>&1 || status=1
      grep -m1 -oE 'PASS|FAIL' "$out/report.md" 2>/dev/null || echo "no report (see $out/run.log)"
      ;;
    tracker_compare)
      "$python" "$repo/python/tracker_compare.py" --out "$out" --corners "$root/cache/corners" > "$out/run.log" 2>&1 \
        || status=1
      tail -2 "$out/run.log"
      ;;
    *)
      MSCKF_GATE_OUT="$run" MSCKF_BUILD="$build" "$python" -m "msckf_gates.$b" > "$out/run.log" 2>&1 || status=1
      grep -m1 -oE 'PASS|FAIL' "$out/report.md" 2>/dev/null || echo "no report (see $out/run.log)"
      ;;
  esac
done

echo
"$python" "$repo/python/benchmark_compare.py" "$run" "${baseline[@]}" || status=1
exit $status
