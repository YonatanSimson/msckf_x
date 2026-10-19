# msckf_x

A monocular camera + IMU **Multi-State Constraint Kalman Filter** in C++17, written from the papers in gated stages. The target is slow, planar, often-stopping ground vehicles (a survey cart) as well as handheld and drone motion, with fisheye and >180° lenses supported from the start.

**License:** GPL-3.0 (`LICENSE`); third-party components and their licenses are listed in `THIRD_PARTY.md`.

**Clean-room:** the math comes from the papers (MSCKF, MSCKF 2.0, the OpenVINS docs, sqrtVINS, Solà). GPL implementations (OpenVINS, VINS-Mono, ORB-SLAM3) are never opened. Everything linked into the filter is permissively licensed (Eigen, Sophus, OpenCV, basalt-headers, yaml-cpp, apriltag). The full design, decisions and running log are in [`mono-msckf-plan.md`](docs/mono-msckf-plan.md).

## Status

Each stage must pass a metric gate, on simulation and on real data, before the next stage starts.

| Stage | Scope | State |
|---|---|---|
| 0 | Camera models, simulator, evaluation harness | **Done**: reprojection gate passes |
| 1a | Static initialization | **Done** |
| 1 | Feature tracking (vendored Basalt patch flow, KLT baseline) | Tracker gate **fails** on TartanAir checkerboard aliasing; deferred |
| 1b | Dynamic initialization (sqrtVINS feature-less linear solve + refinement) | **Done**: gate passes |
| 2 | Front end: gyro-aided tracking, RANSAC on bearings | In progress |
| 3 | MSCKF back end: FEJ, online extrinsic and time offset, nullspace update | Core implemented; simulation gate **fails** (see below) |
| 4 | Pose graph and loop closure (GTSAM iSAM2) | Not started |
| 5 | Optional GNSS factors | Not started |

**Stage 3, current results (simulation, 50 runs × 60 s, 1 px noise):**
- **Handheld motion:** consistent, with orientation / position NEES inside the χ² band in 97 / 100 % of epochs. Final error is 4.9 cm, at 3.3 ms per image.
- **Fast rotation:** calibration converges (0.5 mm, 0.007°, 0.014 ms), but position is inconsistent.
- **Open question:** FEJ raises position NEES where the no-FEJ filter doesn't, which is the opposite of what the theory predicts. The next step is the observability check (numerical rank and ‖O·N‖ with FEJ on and off).

## What's in it

| Module | Contents |
|---|---|
| `camera/` | Pinhole + radtan, Kannala-Brandt, Double Sphere, EUCM, unified (omni) + radtan behind one interface, cross-checked against Basalt |
| `imu/` | RK4 propagation, on-manifold preintegration (cross-checked against GTSAM) |
| `init/` | Static init; feature-less dynamic init with landmark refinement, keyframes, scale gate and retry |
| `vision/` | Vendored Basalt patch optical flow, minimal KLT tracker, bearing RANSAC |
| `geometry/` | Anchored inverse-depth triangulation on bearings (works past 90° off-axis) |
| `filter/` | MSCKF state, closed-form Φ from preintegration with FEJ, cloning with time offset, nullspace-projected update with χ² gating, QR compression, Joseph form |
| `sim/` | Spline trajectories (handheld, fast rotation, car, pushed cart) and synthetic IMU + features |
| `eval/` | ATE / RE with alignment, NEES, timing, reports |
| `calib/` | AprilGrid detection, Kalibr calibration I/O, per-image board pose |

**Conventions:** Hamilton quaternions, body-to-global rotation R_GI, right (body-frame) perturbation, z-up gravity, t_imu = t_cam + t_d, SI units. A unit test pins each of these down.

## Build

You need CMake ≥ 3.24, Ninja, Eigen ≥ 3.4 and OpenCV 4. Sophus, basalt-headers, GoogleTest, apriltag and yaml-cpp are fetched by CMake.

```bash
cmake -S . -B build -G Ninja
cmake --build build
(cd build && ctest --output-on-failure)
```

The GTSAM preintegration cross-check test is optional. Run `scripts/deps/build_gtsam.sh` once and the test is built automatically from then on.

## Data, gates and benchmarks

```bash
scripts/data/fetch_euroc.sh        # also: fetch_tumvi.sh, fetch_tartanair.sh, fetch_kitti_raw.sh, fetch_kitti_flow.sh
pip install -e python/             # gate checkers and viewers (see python/README.md)
scripts/run_gates.sh               # build, unit tests, then every stage gate → gate_out/<gate>/report.md
scripts/run_benchmarks.sh          # dated run with manifest, compared against the previous run for regressions
```

Datasets used: EuRoC (all 11 sequences), TUM-VI (fisheye at about 195°), UZH-FPV (Snapdragon, `python/uzh_fpv_to_euroc.py`), LaMAria (Project Aria: the undistorted pinhole export, `python/lamaria_to_euroc.py`, and the raw Fisheye624 SLAM cameras, `python/lamaria_fisheye_to_euroc.py`), TartanAir, KITTI raw (diagnosis only), plus the built-in simulator for NEES, observability and calibration convergence.

Baselines: OpenVINS and sqrtVINS run as unmodified binaries (`scripts/baselines/run_baseline.sh`, bags from `python/euroc_to_rosbag.py`, benchmark `stage3_baselines`) from clones under `~/work/non_commercial/`, a directory nothing in this repository reads code from; the published EuRoC numbers are in `python/msckf_gates/baselines.py`.

## References

- Mourikis & Roumeliotis, ICRA 2007 (MSCKF)
- Li & Mourikis, IJRR 2013 (MSCKF 2.0)
- Geneva et al., ICRA 2020 (OpenVINS)
- Peng et al., T-RO 2025 (sqrtVINS)
- Usenko et al., 3DV 2018 (Double Sphere)
- Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017

The full list is in the appendix of [`mono-msckf-plan.md`](docs/mono-msckf-plan.md).
