#!/usr/bin/env bash
# Builds GTSAM (BSD) as a test-only reference for the IMU preintegration
# cross-check (tests/test_preintegration_gtsam.cpp). Nothing from it is linked
# into the msckf libraries.
#
#   scripts/deps/build_gtsam.sh [-s SRC] [-p PREFIX]
#
# SRC: a GTSAM checkout (default ../gtsam next to this repo), checked at the
# pinned commit below. PREFIX: install prefix (default build-gtsam/install in
# this repo). Then configure msckf_x with -DGTSAM_DIR=PREFIX/lib/cmake/GTSAM.
#
# Options match this project's build so the two can share Eigen objects:
# system Eigen, no -march=native (alignment ABI), no Boost or TBB.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
src="$(cd "$repo/.." && pwd)/gtsam"
prefix="$repo/build-gtsam/install"
pinned=a74146dbe
while getopts "s:p:h" opt; do
  case "$opt" in
    s) src="$OPTARG" ;;
    p) prefix="$OPTARG" ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

head=$(git -C "$src" rev-parse --short=9 HEAD)
if [ "$head" != "$pinned" ]; then
  echo "GTSAM at $src is at $head, expected $pinned (git -C $src checkout $pinned)" >&2
  exit 1
fi

build="$repo/build-gtsam/build"
mkdir -p "$repo/build-gtsam"
exec > >(tee "$repo/build-gtsam/build.log") 2>&1
cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_INSTALL_RPATH="$prefix/lib" \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF \
  -DGTSAM_USE_BOOST_FEATURES=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF
cmake --build "$build"
cmake --install "$build"
echo "installed to $prefix; configure msckf_x with -DGTSAM_DIR=$prefix/lib/cmake/GTSAM"
