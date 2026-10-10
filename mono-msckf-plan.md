# Mono MSCKF from scratch — implementation plan

Oct 5, 2026 · Yonatan

## Status

Updated as work progresses. Decisions that changed a section below are recorded in that section.

| Date | Stage | Done | Commit |
| --- | --- | --- | --- |
| 2026-10-10 | 0 | Build skeleton (CMake, FetchContent for Sophus/GoogleTest), conventions tests, four camera models | a05b88b |
| 2026-10-10 | 0 | Basalt cross-check, simulator, RK4 nominal propagation | 2f44cf0 |
| 2026-10-10 | 0 | Evaluation harness (`msckf_eval`) | 3ee5177 |
| 2026-10-10 | 0 | Omni-radtan model (shared radtan code), AprilGrid detector and `calib_reproj` (`msckf_calib`), TUM-VI reprojection gate, `python/` tools. Stage 0 complete | b1883a8 |
| 2026-10-10 | 1a | Static initialization, `static_init_eval`, EuRoC IMU reader; gate scripts (`scripts/`, `python/msckf_gates/`) | (pending) |

Stage 0 gate status:

| Gate | Target | Result |
| --- | --- | --- |
| Camera round trip, every pixel | < 1e-6 px | ≤ 2e-12 px, all models |
| Camera Jacobians vs finite differences | < 1e-6 rel. | ≤ 4e-9 |
| Cross-check vs Basalt | < 1e-9 px | ≤ 2.3e-13 px |
| Reprojection on TUM-VI, published calibrations | RMSE after Kalibr's 4σ filter ≤ published RMSE (0.093–0.107 px per model and camera), ≤ 5% corners removed, every radius band ≤ published | pinhole-equi 0.083 / 0.087 px, omni-radtan 0.083 / 0.085 px (cam0 / cam1), 2.9% removed |
| Simulated IMU integration, 60 s | < 1e-6 m, < 1e-8 rad | ≤ 7e-8 m, ≤ 1e-11 rad |

Stage 1 gate status:

| Gate | Target | Result |
| --- | --- | --- |
| 1a static init, simulation: NEES (200 Monte Carlo runs) | tilt, gyro bias and tilt + b_g + b_a inside the 95% band | 1.92 / 2.93 / 7.82 (bands [1.73, 2.29] / [2.67, 3.35] / [7.46, 8.56]) |
| 1a static init, EuRoC MH_01–05: gravity direction | < 1° on every stationary window | max 0.77–0.95° per sequence |

**How to reproduce.** Every gate result above comes from committed scripts, never ad-hoc commands:

- `scripts/data/fetch_tumvi.sh` and `scripts/data/fetch_euroc.sh` download, verify and extract the datasets.
- `scripts/run_gates.sh` builds, runs the unit tests and every gate checker (`python/msckf_gates/`, also installed as `gate-<name>` commands). Each gate writes its outputs and `gate_out/<gate>/report.md`, with a PASS/FAIL table and the ungated diagnostics.

Data lives in `~/data` ($MSCKF_DATA; overflow `/mnt/c/Users/user/data`, $MSCKF_DATA_LARGE), never in the repo:

- `~/data/tumvi/`: calib-cam1 and room1 (EuRoC format, 512×512), calib-cam8 (bag) and the published Kalibr results.
- `/mnt/c/Users/user/data/euroc/`: Machine Hall MH_01–05 (ASL format), from the ETH Research Collection bundle (record 20.500.11850/690084; the old robotics.ethz.ch links are dead).

## Scope and rules

Build a monocular camera + IMU MSCKF in C++17, clean-room from papers only, in gated stages: initialization, front end, back end. No stage starts until the previous one passes its metric gate on both simulation and real data.

**Primary sources (math only, no code):**

- OpenVINS documentation and its ICRA 2020 paper (Geneva et al.): propagation, FEJ, MSCKF update, calibration Jacobians.
- Li & Mourikis, MSCKF 2.0 (IJRR 2013): closed-form transition matrix, observability, FEJ.
- Geneva & Huang, OpenVINS state-initialization report, and the sqrtVINS paper (Peng, Chen, Huang): static and dynamic initialization.
- Usenko et al., Double Sphere camera model (3DV 2018), and Kannala & Brandt (PAMI 2006) for camera models.
- Trawny & Roumeliotis, indirect Kalman filter report: reference for the JPL error-state algebra being translated.

**Clean-room rule:** GPL code (OpenVINS, VINS-Mono, ORB-SLAM3) is never opened while implementing. Permissively licensed libraries (Basalt, GTSAM, COLMAP, OpenSfM) may be read for mechanics: data structures, numerical tricks, test patterns.

**Out of scope for v1:** SLAM features kept in the state, and the square-root filter form. Loop closure (Stage 4) and GNSS (Stage 5) come only after VIO passes its gates; until then, the system runs fully GNSS-denied.

## Conventions: Hamilton, body-to-global, right perturbation

Decision: Hamilton quaternions, store the body-to-global rotation R_GI, and perturb on the right (body frame). Every tool in the stack (Eigen, Sophus, GTSAM, Ceres, Basalt, COLMAP) is Hamilton, so conversions vanish at every library boundary. The right perturbation makes δθ exactly the same body-frame quantity as JPL's δq ⊗ q̂, so the papers' Jacobians carry over with one substitution.

$$
R_{GI} = \hat R_{GI}\,\mathrm{Exp}(\delta\theta) \quad\Longleftrightarrow\quad \bar q_{JPL} = \delta\bar q \otimes \hat{\bar q}_{JPL}, \qquad R_{GI} = C({}^I_G\bar q)^\top
$$

| Quantity | JPL papers (OpenVINS, MSCKF 2.0) | This implementation |
| --- | --- | --- |
| Attitude state | ⁱ_G q̄, matrix C maps G → I | R_GI (Sophus SO3), maps I → G |
| Attitude error | q̄ = δq̄ ⊗ q̂, δθ in I | R = R̂ Exp(δθ), δθ in I (same vector) |
| Any C(q̂) in a Jacobian | C | R̂_GIᵀ |
| Any C(q̂)ᵀ in a Jacobian | Cᵀ | R̂_GI |
| Extrinsic | ᶜ_I q̄, ⁱp_C | R_IC, p_IC (camera pose in IMU frame) |
| Clone | camera or IMU pose | IMU pose (R_GI, p_GI) at image time |
| Composition | ᶜ_A q̄ = ᶜ_B q̄ ⊗ ᴮ_A q̄ | q_AC = q_AB ⊗ q_BC |

Fixed conventions: gravity g = [0, 0, −9.81] m/s² in G (z up), accelerometer measures specific force aₘ = R_GIᵀ(a − g) + b_a + n_a, time offset t_imu = t_cam + t_d, all units SI.

Pitfall to test for on day one: Eigen's Quaternion constructor takes (w, x, y, z) but stores (x, y, z, w). One unit test round-trips every rotation through quaternion, matrix and Exp/Log.

## Stage 0: camera models, simulator, evaluation tooling

Stage 0 builds what every later metric depends on: one camera interface, a simulator with ground truth for consistency tests, and a fixed evaluation harness. Expect about 2 weeks.

### Camera model interface

Every model implements the same four operations, all in double precision:

1. `project(p_c)`: pixel, plus a validity flag.
2. `project` Jacobians: with respect to the 3D point (2×3) and the intrinsics (2×n), the latter for optional online intrinsic calibration.
3. `unproject(uv)`: a unit bearing vector, never a point on the z = 1 plane, so rays beyond 90° stay representable.
4. `valid(uv)` and `valid(p_c)`: the model's own validity region, used to mask detection and reject projections.

The back end and triangulation work on unit bearings and pixel residuals only. They never assume a pinhole, which is what makes the >180° lens work without special cases.

| Model | Use | Parameters | Unproject | Notes |
| --- | --- | --- | --- | --- |
| Pinhole + Brown-Conrady (radtan) | Narrow lenses, EuRoC | fx fy cx cy k1 k2 p1 p2 (k3) | Iterative, Gauss-Newton | Invalid behind the camera; distortion diverges near 90° |
| Kannala-Brandt (KB4, equidistant) | Wide fisheye up to ~195° | fx fy cx cy k1–k4 | Newton on θ | Angle-based polynomial, handles θ > 90° |
| Double Sphere (DS) | Fisheye beyond 180° | fx fy cx cy ξ α | Closed form | Cheap both ways; explicit validity condition |
| Extended Unified (EUCM) | Optional, wide fisheye | fx fy cx cy α β | Closed form | Same interface; useful as cross-check |
| Unified + radtan (Mei, Kalibr omni-radtan) | Fisheye beyond 180° | fx fy cx cy ξ k1 k2 p1 p2 | Newton on radtan, then closed-form lift | TUM-VI publishes it; radtan code shared with pinhole-radtan |

Double Sphere projection, per Usenko et al. (3DV 2018), with the point p = (x, y, z):

$$
d_1 = \sqrt{x^2+y^2+z^2},\quad d_2 = \sqrt{x^2+y^2+(\xi d_1 + z)^2},\quad u = f_x\frac{x}{\alpha d_2 + (1-\alpha)(\xi d_1 + z)} + c_x
$$

The same denominator applies to v. The validity condition on z and the closed-form unprojection are taken from the paper and checked against Basalt's BSD camera-model implementation as a reference.

Radtan validity (decided in implementation): the valid region is the disk where the 2D distortion Jacobian's determinant stays positive along every azimuth, shrunk by 1%. Radial monotonicity alone isn't enough with tangential terms: a test found two rays projecting to the same pixel inside it. Basalt's radtan8 unprojects some pixels that its own projection then rejects; ours is self-consistent.

**Stage 0 camera metrics (all must pass):**

- Round trip unproject → project over every pixel inside the valid region: max error < 1e-6 px, including rays past 90° for KB4 and DS.
- Analytic Jacobians vs central finite differences on 10⁵ random valid points: max relative error < 1e-6.
- Cross-check against Basalt's camera models on the same parameters and points: < 1e-9 px difference.
- Reprojection error on the TUM-VI calibration sequence with TUM-VI's published calibrations held fixed (`pinhole-equi` → KB4, `omni-radtan` → unified + radtan), board pose fitted per image: after Kalibr's corner outlier rule, RMSE ≤ 0.105 px with ≤ 5% of corners removed, and no radial trend in the inlier RMSE. This checks the models and the Kalibr parameter mapping on real data; a wrong model or mapping is off by pixels, not hundredths.
  - Reference: TUM-VI's published camera-IMU result is mean 0.094 px, std 0.047 px of the error norm after Kalibr's outlier removal, so RMSE = √(0.094² + 0.047²) ≈ 0.105 px. Our setting is easier (free pose per image instead of poses tied to the IMU trajectory) and the sequence differs, so we expect to land at or below it, not on it.
  - Kalibr's outlier rule (`kalibr_calibrate_cameras`, BSD-3): drop a corner when either residual component exceeds 4σ of that component over all corners; refit and repeat. Applied in `calib_reproj` (converges in 2–4 rounds).
  - Changed during implementation: the gate was "RMS within 5% of the published value, for the KB4 and DS calibrations". TUM-VI publishes `pinhole-equi` and `omni-radtan`, not DS, and its only published error is the camera-IMU run's, so an exact reference would need a Kalibr re-run. That was dropped as out of proportion: the Basalt cross-check already proves the implementations (DS and EUCM included), and this gate only has to show the published calibrations work through our code. Omni-radtan was added as a model so both published calibrations can be used. An interim version used the median (0.085 px), but that compared our unfiltered median with Kalibr's filtered mean; RMSE on the same filtering is the right comparison.
  - Result (calib-cam1, all 435 frames, ~14,600 corners per camera): all corners RMSE 0.34 px (a 3% tail); after the 4σ rule, cam0 RMSE 0.083 px for both models (mean 0.075, median 0.071, std 0.037), cam1 0.087 / 0.085 px. Outliers grow toward the image edge (1% within 100 px of the center, 9% beyond 230 px) identically for both models, while the inlier RMSE stays 0.08–0.09 px at every radius: noisier edge corners (blur, small foreshortened tags), not model error.
  - Tool: `tools/calib_reproj` (C++, `msckf_calib` library) reads a Kalibr camchain and target yaml and writes `frames.csv` and `corners.csv` for plotting from `python/`.
  - Target: AprilGrid 6×6, tag 0.088 m, spacing 0.3. Detector: AprilRobotics apriltag 3.4.2 (BSD-2) with tag36h11 re-described for Kalibr's 2-bit black border (width_at_border 10, data bits shifted by one), then Kalibr's `cornerSubPix` (2×2 window, 1.5 px² displacement limit, 4 px border, ≥ 4 tags).
  - apriltag fits quads to thresholded blob outlines. On an AprilGrid each tag's outer border meets the corner squares at X-junctions, so the blob is not a quad: 2/36 tags per frame. A 2×2 grayscale dilation before detection opens the junctions and keeps the ~2.5 px bits decodable: 32–36/36. Corners are refined on the original image.
  - Kalibr's own detector, `ethz_apriltag2`, is Kaess's C++ port of the original AprilTags (gradient edges → segments → quads) and handles the 2-bit border natively. Its package.xml says BSD, but the sources carry no headers and the upstream Kaess library is LGPL-2.1, so under the clean-room rule its sources are not read.

### Simulator

A simulator with exact ground truth is the only way to measure consistency (NEES), so it's built before the filter.

- **Trajectory:** a C² cubic B-spline on SE(3) through random or scripted control poses, giving analytic angular velocity and acceleration at any time.
  - Implemented as a split SO(3) × R³ cumulative uniform cubic B-spline: exact world acceleration for the IMU model, with closed-form body angular velocity and acceleration. The curve approximates its control poses rather than passing through them.
- **IMU:** 200–400 Hz, white noise plus bias random walk with configurable densities, matching the IMU's datasheet or Allan variance.
- **Landmarks:** points on walls and a ground plane; the ground plane matters because the cart sees mostly ground.
- **Camera:** any of the models above, 1 px pixel noise, optional time offset and rolling-shutter off for v1.
- **Scenarios:** handheld (rich excitation), cart (slow, planar, long constant-velocity runs, stops), and fast rotation.

Simulator gate: noiseless IMU integrated over 60 s matches the spline to < 1e-6 m and < 1e-8 rad. This validates both the simulator and the future propagation code.

- The gate uses exact IMU readings at RK4's midpoints, from the continuous-time model. Fast rotation (peak 6.3 rad/s) runs the gate at 800 Hz: at 400 Hz it lands at 1.07e-6 m, purely from RK4 step size. A separate test confirms 4th-order convergence (error ratio 16.0 per halving of the step) in every scenario.
- Noted for Stage 3: with discrete samples only, linearly interpolated midpoints cost 5 mm (handheld) to 15 cm (fast rotation) over 60 s noiseless. That's far below IMU noise, but it's the propagation floor.

### Evaluation harness

One command runs a config over a dataset list and writes a fixed report:

- **Absolute trajectory error (ATE):** RMSE after 4-DOF alignment (position + yaw), since that's the VIO gauge.
- **Relative error (RE):** translation % and rotation deg/m over segment lengths, rpg_trajectory_evaluation style. Errors are divided by each segment's actual ground-truth path length, not the nominal one, which removes an upward bias of up to one sample spacing.
- **NEES:** pose NEES on simulation over 50 Monte Carlo runs, with the χ² 95% band.
- **Timing:** per-module latency and CPU load on the Jetson Orin, median and 99th percentile.

Implemented as `msckf_eval`: dataset list (`name gt est [tum|euroc]`) in, `report.md` and `metrics.csv` out. ATE alignment can be posyaw, se3 or sim3, optionally fitted on the first N seconds (the Stage 1 scale metric). Until an estimator exists, it evaluates trajectory files already on disk; running the estimator per dataset is added with Stage 1.

## Stage 1: initialization

Stage 1 delivers an initial IMU state (attitude, velocity, biases) with a covariance the filter can trust, from either a static start or a moving one. It runs standalone on recorded data with a simple feature tracker, so it can be built and measured before the real front end exists. Expect about 3 weeks.

Assumptions, as in both source papers: camera-IMU extrinsics and time offset are known from offline calibration, and IMU biases start reasonably close (from a prior calibration or a static window).

### 1a. Static initialization

This is the primary path for the cart, which almost always starts stationary.

1. Detect stationarity over a ~1 s window: accelerometer standard deviation below a threshold and median feature disparity below ~1 px. Both are needed, since a slow constant-velocity cart passes the IMU test alone.
2. Gravity direction from the mean specific force sets roll and pitch; yaw is set to 0 (it's unobservable).
3. v = 0, p = 0, b_g = mean gyro, b_a from prior calibration.
4. Covariance: roll and pitch from accelerometer noise over the window, gyro bias from gyro noise over the window, yaw and position set to a small value since they define the gauge.

Implementation notes (`include/msckf/init/static_init.h`, `tools/static_init_eval`):

- Output is `ImuInitState`: the IMU state and the 15-dim error covariance in the filter's order (δθ, δp, δv, δb_g, δb_a; `imu_err` indices), shared with dynamic init.
- Yaw = 0 means the global x axis is the horizontal projection of the IMU x axis.
- Covariance is the first-order propagation of the window-averaged noise (σ²/T) and the b_a prior: δθ_tilt = ⌊z×⌋(n̄_a + δb_a)/|f|, so tilt and b_a are correlated and that cross term is in P. The biases also get the random walk between the window mean and the window end, σ_w² T / 3: tilt and b_a are otherwise almost perfectly correlated, and leaving it out made the joint NEES overconfident (8-dof NEES 9.3 vs band [7.5, 8.6]).
- Stationarity: accelerometer std (√trace of the sample covariance) ≤ 0.1 m/s², tracker median disparity ≤ 1 px when available, and |f| within 0.3 m/s² of g. Disparity is an optional input until the KLT tracker exists.
- Simulation (200 Monte Carlo runs, random tilt up to 34°, b_a drawn from the prior): mean NEES inside the 95% band for tilt (2 dof), gyro bias (3) and tilt + b_g + b_a (8), checked over four seeds; exact (1e-9) without noise.
- Real data, TUM-VI: no static starts. room1 (and calib-cam1) begin handheld: the lowest 1 s accelerometer std is 0.13 m/s² and all windows are correctly refused, so TUM-VI tests dynamic init only.
- Real data, EuRoC MH_01–05 (gate `stage1a_static_init`): the sequences do not start at rest either. The first ~20 s are a handheld IMU-excitation motion, then the MAV sits on the ground with motors off for a few seconds before take-off, which is where the stationary windows are (2–19 per sequence). Tilt error 0.69–0.89° mean, 0.95° max. Almost all of it is accelerometer bias: with EuRoC's ground-truth b_a as the prior it drops to 0.05–0.13°. Static init cannot separate tilt from b_a, so without a bias calibration ~1° is the floor for an ADIS16448-class IMU.
- Accelerometer bias prior for EuRoC: σ = 0.1 m/s² per axis (uncalibrated MEMS; EuRoC's ground truth has |b_a| ≈ 0.15 m/s²). With the code default 0.05 (which assumes a prior calibration) the tilt NEES on MH_01 was 9.2 (2 dof), overconfident; with 0.1 it is 1.4–2.3.
- Gyro bias on EuRoC (not gated): our estimate is repeatable across stationary windows (0.06–0.31 mrad/s spread) but sits 1.1–1.5 mrad/s from EuRoC's ground-truth b_w in every sequence (NEES 47–80). EuRoC's biases come from its own estimator rather than a measurement, and the mean gyro of a stationary window is the most direct estimate there is, so this is treated as a reference discrepancy to revisit in Stage 3 (the filter will estimate b_g too), not an initializer error.

### 1b. Dynamic initialization

For starts in motion, follow the two-step design in sqrtVINS (Peng, Chen, Huang, TRO 2025):

1. **Feature-less linear solve.** Recover the first-frame velocity and gravity direction without triangulating any feature. Bearings are rotated into the first IMU frame with gyro integration, and the translation direction between keyframes is the eigenvector with the smallest eigenvalue of the summed epipolar-plane normals. IMU preintegration makes the same displacement linear in (v₀, g); projecting it onto the plane orthogonal to that direction removes the unknown scale, leaving a small linear system in v₀ and g. Implement this faithfully from the paper.
2. **Refinement.** Optimize the IMU state, features and covariance jointly. As in sqrtVINS, this is done with iterated filter updates (EKF form here, not square-root), relinearizing until convergence and computing the covariance once at the end. Features stay in the state with their covariance, so the filter has constraints from the first frame after hand-off. This is the primary path. A batch MLE (Ceres), following the OpenVINS initialization report, is kept only as an offline cross-check on accuracy and covariance.
3. **Excitation gate.** Reject the solve if the smallest singular value of the linear system or the accelerometer excitation is below a threshold, and retry with a longer window. Mono scale is only observable through acceleration, so this gate matters most on the cart.

The paper reports a high success rate with windows as short as 100 ms, tested on Jetson Nano as well as workstations. Treat that as the ceiling, not the v1 target.

**Cart baseline.** sqrtVINS is the only initializer observed to work on the pushed cart, so it's the reference for cart data. Run it as an unmodified binary on the same segments and windows; it's GPL, so its source stays closed under the clean-room rule. On cart data, the Stage 1 gate is matching its success rate.

### Stage 1 metrics

The protocol is adopted from sqrtVINS, so results compare directly with its published monocular numbers. Split each sequence into 10 s segments and, at each segment start, run initialization with windows of 0.1, 0.2, 0.3, 0.5, 0.75 and 1.0 s. A run succeeds only if all three hold:

1. The linear system is well-posed and passes the excitation gate.
2. Refinement converges and produces a valid covariance.
3. VIO position ATE over the following 10 s stays below a threshold, reported at 0.1, 0.3 and 0.5 m.

Run on simulation (50 seeds per scenario), EuRoC and TUM-VI (left camera only), and cart data. This judges initialization by whether the VIO that follows actually works, not just by the initial state. Targets below are proposed starting points.

Gate split (decided at the start of Stage 1): condition 3 needs the filter, which is Stage 3. So the Stage 1 gate covers everything measurable against ground truth at hand-off: conditions 1 and 2, gravity, velocity and scale error, initial NEES and runtime. The success rate with condition 3 (VIO ATE over the next 10 s) is re-run, and gated, once Stage 3 exists; the targets below stay as they are.

Build order: 1a static init (needs only the IMU, plus the tracker's median disparity when available), IMU preintegration (cross-checked against GTSAM's formulas), a minimal KLT tracker on our camera models that feeds bearings and disparity to the initializer (sqrtVINS style; the Stage 2 front end replaces it), then 1b: linear solve, refinement, excitation gate.

| Metric | How | v1 target |
| --- | --- | --- |
| Success rate | Share of segment starts meeting all three conditions, per window length and ATE threshold (condition 3 from Stage 3 on) | ≥ 95% at 0.5 m with a 1 s window on EuRoC and TUM-VI |
| Cart success rate | Same protocol on cart sessions, static and rolling starts | ≥ the sqrtVINS binary on the same segments |
| Time to initialize | Shortest window meeting the success target | ≤ 1 s handheld; static path on cart |
| Gravity direction error | Angle between estimated and ground-truth gravity in the first IMU frame | < 1° |
| Velocity error | Norm vs ground truth | < 0.1 m/s or < 10% |
| Scale error | Sim(3) fit of the first 10 s to ground truth, 100 × (max(s, 1/s) − 1) | < 5% |
| Initial NEES | Roll, pitch, velocity, gyro bias on simulation | Inside the χ² 95% band |
| Runtime | Linear solve and refinement on Orin | Linear < 5 ms, refinement < 50 ms |

Sources: [sqrtVINS, arXiv 2510.10346](https://arxiv.org/abs/2510.10346v1).

## Stage 2: front end

Stage 2 produces long, clean feature tracks on raw distorted images for every camera model, and hands the back end a list of observations per feature. Images are never undistorted: undistortion is impossible beyond 180° and wastes resolution on wide lenses. Expect about 3 weeks.

### Pipeline per frame

1. **Pyramid:** 3–4 levels, optional CLAHE for lighting changes outdoors.
2. **Gyro-aided prediction:** rotate each feature's bearing by the gyro-integrated rotation since the last frame (through the extrinsic), project with the camera model, and use that as the KLT starting point. This is what keeps tracks alive under fast rotation and strong distortion.
3. **KLT tracking:** inverse-compositional Lucas-Kanade on the distorted image, coarse to fine. Basalt's patch optical flow is the reference for mechanics. OpenCV's pyramidal LK is the baseline to beat.
4. **Forward-backward check:** track back to the previous frame and drop tracks that return more than ~0.5 px from where they started.
5. **Geometric outlier rejection:** RANSAC on unit bearings, never on pixels, with angular residuals. With rotation known from the gyro, a 2-point model for translation direction is enough and cheap; 5-point is the fallback. Bearings make this valid past 180°.
6. **Detection:** refill empty cells of a grid (FAST or Shi-Tomasi), masked by the camera model's validity region and a minimum spacing, to a target of 150–250 tracked features.
7. **Track management:** each track keeps its id, timestamps, pixels and bearings. A track is sent to the back end when it ends or reaches the clone-window length. In Stage 2 this lives in a simple container; Stage 2b replaces it with the real map.

The front end also outputs median disparity per frame, which Stage 1 and the zero-velocity update in Stage 3 both use.

### Stage 2 metrics

Measured on simulation (exact ground-truth projections, with injected outliers), EuRoC, TUM-VI and UZH-FPV. Real-data accuracy uses ground-truth poses: the angular epipolar error of each track pair. Targets are proposed starting points.

| Metric | How | v1 target |
| --- | --- | --- |
| Tracking accuracy | Pixel error vs ground-truth projection on simulation | RMSE < 0.3 px |
| Track length | Median frames per track, and share reaching window length | Median ≥ 8 at 20 Hz on EuRoC |
| Coverage | Share of grid cells with a live track; share of tracks with ray angle > 60° on fisheye | > 70% of cells; periphery not empty |
| Outlier rate after RANSAC | Tracks above a ground-truth angular-error threshold | < 2% |
| RANSAC precision and recall | Against injected outlier labels on simulation | Both > 95% |
| Gyro-prediction gain | Track survival on UZH-FPV with vs without prediction | Clear improvement, reported per sequence |
| Runtime | Per frame on Orin at 640×480 | < 8 ms |

## Stage 2b: feature and frame map

Stage 2b builds the bookkeeping layer that owns every frame, feature and observation: the "map". The front end writes into it, and triangulation, the MSCKF update, marginalization, the zero-velocity update and the Stage 4 keyframes all read from it. It comes right after the front end, because its access patterns are only clear once real tracks exist. Expect about 2 weeks.

### Study first: three reference designs

All three are BSD-licensed, so reading them for mechanics is within the clean-room rule. Spend the first few days on a one-page comparison of how each one stores, queries and deletes, then design your own.

- **COLMAP:** an offline SfM scene graph. Each image holds its 2D points with links to 3D points, each 3D point holds a track of (image, point index) pairs, and a separate correspondence graph holds the raw matches. The reference for clean bidirectional links and track merging and splitting.
- **OKVIS2-X:** a keyframe-based factor-graph VIO/SLAM. Observations are keyed by an identifier of (frame, camera, keypoint), and landmarks keep their observation sets. Closest to a live sliding window with keyframes and loop closure, and the newest of the three.
- **Basalt:** a landmark database for a sliding window. Each landmark is hosted in a keyframe as bearing plus inverse distance, observations are grouped by target frame, and there are explicit routines to drop frames and landmarks on marginalization. The reference for fast window-shaped deletions.

### Entities

- **Frame:** id, timestamp, camera id, keyframe flag, link to its clone index in the filter state, and the ids of features observed in it.
- **Feature:** id, status (tracking, lost, used in update, rejected), ordered observations, and an optional anchor frame plus inverse-depth estimate with a quality flag once triangulated.
- **Observation:** frame id, feature id, pixel, unit bearing, and noise.

Ids are monotonic 64-bit integers that are never reused. Structures reference each other by id, never by pointer, so deletion can't leave dangling references.

### Operations

| Operation | Used by | Target cost |
| --- | --- | --- |
| Insert observation | Front end, every frame | O(1) |
| Features seen in a frame | Detection masking, marginalization | O(features in frame) |
| Observations of a feature | Triangulation, MSCKF update | O(track length) |
| Finished or max-length tracks | MSCKF update | O(finished tracks) |
| Tracks touching the oldest clone | Marginalization, before removal | O(features in that frame) |
| Delete feature | After update or rejection | O(track length) |
| Delete frame | Marginalization; also removes its observations | O(features in frame) |
| Remap clone indices | After marginalization shifts the state | O(window) |

### Concurrency

One writer at a time. The front end inserts during a frame; at the frame boundary the back end takes a batch (the tracks to process and the frame to marginalize) as a copy, so the filter never reads while the front end writes. The batch hand-off is also the natural place for logging and replay.

### Stage 2b metrics

| Metric | How | v1 target |
| --- | --- | --- |
| Correctness | Randomized insert, query and delete sequences checked against a naive reference model | 0 mismatches over 10⁶ operations |
| Invariants | Every link valid in both directions; no dangling ids after any deletion | Checked after every operation in tests |
| Agreement with front end | Track lengths and counts in the map equal the front end's own | Exact |
| Latency | All operations for one frame on Orin, 250 features × 11 frames | < 0.5 ms per frame |
| Stress | 2,000 features × 50 frames | < 5 ms per frame |
| Memory | Resident size over a 1 h run | Flat after warm-up |
| Replay | Serialize, reload and re-run the back end from saved batches | Bit-identical estimates |

## Stage 3: back end (MSCKF)

Stage 3 is a standard error-state MSCKF with IMU-pose clones, FEJ, online extrinsic and time-offset calibration, and a zero-velocity update for the cart. Expect about 5 weeks, plus 2 weeks of tuning on real data.

### State

- **IMU:** R_GI, p, v, b_g, b_a, with a 15-dim error state.
- **Calibration:** extrinsic R_IC, p_IC (6), time offset t_d (1), intrinsics optional and model-dependent.
- **Clones:** IMU poses at image times, window N ≈ 11, configurable.

### Propagation

Nominal state is integrated with RK4 over IMU samples. Covariance uses the closed-form transition matrix from MSCKF 2.0, rewritten in this doc's Hamilton convention:

$$
\dot{\delta\theta} = -\lfloor\hat\omega\times\rfloor\delta\theta - \delta b_g - n_g, \qquad \dot{\delta v} = -\hat R_{GI}\lfloor\hat a\times\rfloor\delta\theta - \hat R_{GI}\,\delta b_a - \hat R_{GI}\,n_a, \qquad \dot{\delta p} = \delta v
$$

FEJ: each state keeps its first estimate, and every Φ and H is evaluated there. The FEJ switch is a config flag so its effect can be measured.

### Cloning

Propagate to the image time t_cam + t_d, clone the IMU pose, and augment P with the clone Jacobian. The time-offset column of that Jacobian comes from the current v and ω.

### Feature update

1. **Triangulate** each finished track: anchored inverse depth in the first observing clone, Gauss-Newton on bearings. Reject on low parallax, poor conditioning, or rays pointing away from the point.
2. **Linearize** the pixel residual through the camera model's Jacobian. With ⁱp_f = R_GIᵀ(p_f − p_GI) and ᶜp_f = R_ICᵀ(ⁱp_f − p_IC), the clone blocks are ⌊ⁱp̂_f×⌋ for δθ, −R̂_GIᵀ for position, and R̂_GIᵀ for the feature, each multiplied by R_ICᵀ and the camera Jacobian.
3. **Nullspace projection** with Givens rotations, then a χ² gate at 95% with 2M − 3 degrees of freedom.
4. **Stack and compress** all accepted features with QR, then a Joseph-form update and injection R ← R̂ Exp(δθ).
5. **Marginalize** the oldest clone when the window is full, after using every track that touches it.

### Zero-velocity update

When the IMU and the front-end disparity both say stationary, apply v = 0 as a measurement and stop adding clones. Without it, a stopped cart lets velocity and bias errors grow, since the visual update has no parallax.

### Error parameterization: FEJ vs right-invariant

The target is GNSS-denied or partially denied operation, so yaw and position stay unobservable for long stretches and consistency matters. Both error parameterizations are implemented behind the same interface: standard multiplicative error with FEJ, and right-invariant SE₂(3) error. Only Φ, the clone Jacobian, H and the reset differ; clones, nullspace projection and QR are shared. Keep whichever has the better NEES over long simulated runs; ATE decides only if both are consistent.

### Stage 3 metrics

| Metric | How | v1 target |
| --- | --- | --- |
| Jacobian correctness | Every Φ and H block vs finite differences | Relative error < 1e-6 |
| Observability | Numerical rank and ‖O·N‖ on simulation, FEJ on vs off | 4 unobservable directions with FEJ |
| Consistency | Average NEES of orientation and position over 50 Monte Carlo runs | Inside the χ² 95% band with FEJ |
| Accuracy | ATE (4-DOF aligned) and RE on EuRoC, TUM-VI, UZH-FPV | Within 20% of OpenVINS mono, run as a binary |
| Drift on cart data | Final position error as % of distance, RTK as ground truth | Reported per run; baseline for later work |
| Calibration convergence | Extrinsic and t_d from a perturbed start on simulation | < 1 mm, < 0.1°, < 0.5 ms |
| Robustness | Divergences per hour of data | 0 on all test sequences |
| Runtime | Full pipeline on Orin | Real time at 20 Hz camera, < 30% of one core for the back end |

Running OpenVINS as an unmodified binary for comparison is compatible with the clean-room rule; reading its source is not.

## Stage 4: pose-graph back end and loop closure

Stage 4 adds a GTSAM factor graph on keyframes that removes drift through loop closures, while the MSCKF keeps running unchanged as the odometry source. Expect about 5 weeks.

### Architecture

- **Decoupled:** the filter never receives loop information in v1. The back end publishes a correction T_map_odom, and the corrected pose is that correction composed with the filter's pose. This avoids feeding the same information back twice.
- **Keyframes:** selected from filter clones by parallax, distance and time. Each stores its bearings, triangulated points, loop descriptors, and the filter's covariance at selection.
- **Odometry factors:** 6-DOF between-factors from consecutive keyframes, with covariance from the filter's joint marginal of the two poses. Roll and pitch get prior factors from gravity, which stays observable without GNSS. Overlapping filter windows make consecutive factors slightly correlated; keyframe spacing keeps this small, and it's checked on simulation.
- **Solver:** iSAM2 for incremental updates, with a full batch re-solve after each accepted loop.

### Loop closure

1. **Detection:** global descriptors from the existing VPR stack, or a bag of words as a baseline, against keyframes outside a recent-time window.
2. **Verification:** match local features, then P3P RANSAC on bearings against the old keyframe's triangulated points, so it works on fisheye and >180° lenses. Refine the relative pose and accept on inlier count and reprojection error.
3. **Robustness:** loop factors use a robust kernel or graduated non-convexity, so one false loop can't bend the map.
4. **Scale check:** mono-inertial scale is observable but drifts under weak excitation. Log the scale disagreement at each loop; switch loop factors to Sim(3) only if it proves significant on cart data.

### Stage 4 metrics

| Metric | How | v1 target |
| --- | --- | --- |
| Loop precision and recall | Against ground-truth revisits within a radius | Precision > 99%, recall reported |
| False-loop robustness | Inject 10% false loops on simulation | ATE changes < 5% |
| Drift removal | ATE before vs after on sequences with revisits | Reported per sequence; no sequence worse |
| Latency | iSAM2 update per keyframe on Orin | < 50 ms |
| Memory | Graph and keyframe storage per km | Reported; baseline for long scans |

Test data with revisits: EuRoC Machine Hall, TUM-VI room and corridor sequences, and cart scans with deliberately repeated lines.

## Stage 5: GNSS, whenever it exists

Stage 5 adds GNSS as optional factors in the Stage 4 graph, so the same system handles a scan with full, partial, or no GNSS without a mode switch. Expect about 4 weeks.

### Design

- **GNSS position factors** on keyframes, interpolated to the fix time with VIO odometry, including the antenna lever arm and a GNSS-IMU time offset.
- **Noise from fix quality:** RTK fixed, float and single each get their own covariance, with a robust kernel against multipath near buildings and trees.
- **Frame alignment:** a T_ENU_map variable (yaw and translation, since gravity already fixes roll and pitch). It's added once the GNSS track has enough spatial spread to make yaw observable, then kept across outages.
- **No GNSS at all:** the variable is never created, and the graph is just VIO plus loops.
- **Optional later:** a GNSS update inside the filter for real-time georeferenced output. The intermittent-GPS VIO work from the OpenVINS group (Lee et al., ICRA 2020) is the reference for online alignment there.

The Stage 3 consistency work pays off here: after an outage, an overconfident filter makes the first fixes look like outliers. That's why NIS at re-acquisition is a gate.

### Stage 5 metrics

Cut artificial outages of 30 s, 2 min and 5 min from cart runs with continuous RTK, so ground truth exists through each gap. Then add real denied runs (urban canyon, tree cover).

| Metric | How | v1 target |
| --- | --- | --- |
| Outage drift | Position error at the end of each outage, % of distance traveled | Reported per outage length; baseline for later work |
| Re-acquisition consistency | NIS of the first GNSS factors after each outage | Inside the χ² 95% band |
| Correction jump | Pose change at re-acquisition | Matches the drift, no overshoot |
| Alignment accuracy | Yaw and translation of T_ENU_map vs ground truth | < 0.5° yaw after alignment |
| Multipath rejection | Injected GNSS outliers on simulation | Rejected without moving the map |

## Datasets

Use the left camera only from stereo datasets. Together these cover every camera model, from rich handheld excitation to slow, planar, cart-like motion. Specs below are from memory and approximate; check each dataset's page before relying on a number.

| Dataset | Camera model exercised | Motion | Ground truth | Why it's here |
| --- | --- | --- | --- | --- |
| Own simulator | All four models | Handheld, cart, fast rotation | Exact | NEES, observability, calibration convergence |
| EuRoC MAV | Pinhole + radtan | Drone, indoor | Vicon / laser tracker, full trajectory | Standard accuracy benchmark, easy comparison |
| TUM-VI | KB4 and unified + radtan (~195° lens) | Handheld, indoor and outdoor | Mocap in the room, start/end only elsewhere | The >180° test; published `pinhole-equi` and `omni-radtan` calibrations |
| TUM-VIE | Wide-angle frame cameras; check the calibration model before use | Handheld and head-mounted, sports, HDR, indoor and outdoor | Mocap at start and end only | Stage 2–3 robustness: fast motion, HDR, drift at sequence end. Not for Stage 1 metrics |
| UZH-FPV | Equidistant fisheye | Aggressive drone flight | Laser tracker | Fast rotation, motion blur, gyro-aided tracking |
| Rosario | Pinhole | Slow agricultural robot, planar | RTK GPS | Closest public match to the cart's motion |
| Exodigo cart (own) | ZED-X left (pinhole), one Insta360 lens (DS) | Slow, planar, frequent stops | RTK GPS | The target system |

Record the cart sessions with a calibration sequence (Kalibr, Basalt calibration) at the start of each campaign, and include stops and turns on purpose so initialization and the zero-velocity update are exercised.

## Packages

Everything linked into the filter is permissively licensed, so the result stays commercially usable. Licenses are from memory; confirm each before shipping. The core library is ROS-free, with an optional ROS 2 wrapper.

**Linked into the filter:**

| Package | Role | License |
| --- | --- | --- |
| Eigen | Linear algebra, QR, Givens | MPL-2.0 |
| Sophus | SO(3)/SE(3), Exp/Log, Hamilton | MIT |
| Ceres Solver | Batch MLE refinement in initialization | BSD-3 |
| OpenCV | Image I/O, pyramids, FAST; LK only as a baseline | Apache-2.0 (4.5+) |
| yaml-cpp, spdlog | Config, logging | MIT |
| GoogleTest | Unit tests, Jacobian checks | BSD-3 |
| Rerun or Pangolin | Live visualization of tracks, clones, covariance | MIT / Apache-2.0 |

From Stage 4, GTSAM (BSD-3) is also linked, for the iSAM2 pose graph and the GNSS factors.

**Tools, used outside the filter:**

- **Kalibr** and **Basalt calibration:** camera-IMU calibration for our own rigs (cart), all models including DS and EUCM. Kalibr would run in Docker (ROS 1 Noetic image from its own Dockerfile), since Ubuntu 24.04 has no ROS 1; not needed for public datasets, which ship calibrations.
- **basalt-headers** (BSD-3): test-only dependency for the Stage 0 camera cross-check, pinned by hash; never linked into the filter.
- **evo:** ATE and RE plots. It's GPL, which is fine for a standalone evaluation tool that isn't linked.
- **COLMAP:** offline SfM poses as orientation ground truth on cart data, where RTK only gives position.

**Reference implementations for mechanics (read, don't copy):**

- **Basalt:** camera model formulas and validity checks, patch optical flow, IMU preintegration.
- **GTSAM:** preintegration math and Lie-group utilities, as a cross-check for the initialization factors.
- **COLMAP:** robust triangulation checks and RANSAC infrastructure.
- **OpenSfM:** fisheye camera handling and bearing-based geometry.

Never opened during implementation: OpenVINS, VINS-Mono, ORB-SLAM3 source (all GPL).

## Milestones and gates

The plan runs about 29 weeks for one engineer: about 20 to working VIO, then loop closure and GNSS, with each stage gated on its own metrics before the next begins.

| Stage | Duration | Scope | Gate to pass before the next stage |
| --- | --- | --- | --- |
| 0 · Camera models, simulator, evaluation harness | ~2 weeks | Pinhole, KB4, Double Sphere, EUCM, unified + radtan behind one interface | Round trip < 1e-6 px, Jacobians < 1e-6, simulated IMU integration exact |
| 1 · Initialization | ~3 weeks | Static path for the cart, sqrtVINS-style dynamic path | ≥ 95% success at 0.5 m ATE, gravity < 1°, scale < 5%, initial NEES in band |
| 2 · Front end | ~3 weeks | Gyro-aided KLT on raw images, RANSAC on bearings | Tracking < 0.3 px, outliers < 2%, < 8 ms per frame on Orin |
| 2b · Feature and frame map | ~2 weeks | Study COLMAP, OKVIS2-X, Basalt, then build an id-based store | 0 mismatches vs reference model, < 0.5 ms per frame, flat memory |
| 3 · MSCKF back end and tuning | ~5 weeks + 2 tuning | FEJ vs invariant error, online calibration, zero-velocity update | VIO done: NEES in band, ATE within 20% of OpenVINS mono, real time on Orin |
| 4 · Pose graph and loop closure (GTSAM) | ~5 weeks | Decoupled iSAM2 graph, VPR loops verified by P3P on bearings | Loop precision > 99%, robust to 10% false loops, < 50 ms per keyframe |
| 5 · GNSS: full, partial or none | ~4 weeks | Optional position factors, ENU alignment, fix-quality noise | Done: NIS in band at re-acquisition, yaw alignment < 0.5° |

Gate thresholds are the proposed v1 targets from each stage's metrics table. Re-baseline them after the first full run on cart data.

## Appendix: papers

Grouped by where each is used. Venues and years are from memory; verify before citing formally.

**Core (requested)**

1. Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation," ICRA 2007. The original MSCKF.
2. Li & Mourikis, "High-precision, consistent EKF-based visual-inertial odometry," IJRR 2013. MSCKF 2.0: closed-form Φ, observability, FEJ, online extrinsics.
3. Geneva, Eckenhoff, Lee, Yang & Huang, "OpenVINS: A Research Platform for Visual-Inertial Estimation," ICRA 2020, plus the OpenVINS documentation derivations.
4. Peng, Chen, Wu & Huang, "sqrtVINS: Robust and Ultrafast Square-Root Filter-based 3D Motion Tracking," T-RO 2025 (arXiv 2510.10346). Feature-less dynamic initialization and iterated refinement.
5. Usenko, Demmel, Schubert, Stückler & Cremers, "Visual-Inertial Mapping with Non-Linear Factor Recovery," RA-L 2020. Basalt: patch optical flow, marginalization, factor recovery for Stage 4.

**Conventions and error state (Stage 3)**

6. Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation," tech report, Univ. of Minnesota, 2005. JPL quaternion and error-state reference.
7. Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication," Aerospace 2018. JPL vs Hamilton translation.
8. Barrau & Bonnabel, "The Invariant Extended Kalman Filter as a Stable Observer," IEEE TAC 2017. Right-invariant SE₂(3) error.
9. Wu, Zhang, Su, Huang & Dissanayake, "An Invariant-EKF VINS Algorithm for Improving Consistency," IROS 2017. Invariant MSCKF.
10. Li & Mourikis, "Online temporal calibration for camera-IMU systems," IJRR 2014. Time-offset state.

**Initialization and degenerate motion (Stage 1)**

11. Geneva & Huang, OpenVINS state initialization technical report. Static and Dong-Si dynamic initialization, MLE cross-check.
12. Dong-Si & Mourikis, "Estimator initialization in vision-aided inertial navigation with unknown camera-IMU calibration," IROS 2012. The baseline sqrtVINS compares against.
13. Wu, Guo, Georgiou & Roumeliotis, "VINS on Wheels," ICRA 2017. Unobservable scale under ground-vehicle motion; the cart's core problem.
14. Yang, Geneva, Eckenhoff & Huang, "Degenerate Motion Analysis for Aided INS with Online Spatial and Temporal Sensor Calibration," RA-L 2019. Which motions starve calibration states.
15. Forster, Carlone, Dellaert & Scaramuzza, "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry," T-RO 2017. Preintegration for initialization and the Stage 4 graph.

**Camera models and front end (Stages 0 and 2)**

16. Usenko, Demmel & Cremers, "The Double Sphere Camera Model," 3DV 2018.
17. Kannala & Brandt, "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses," PAMI 2006.
18. Baker & Matthews, "Lucas-Kanade 20 Years On: A Unifying Framework," IJCV 2004. Inverse-compositional KLT.
19. Troiani, Martinelli, Laugier & Scaramuzza, "2-Point-based Outlier Rejection for Camera-IMU Systems with Applications to Micro Aerial Vehicles," ICRA 2014. Gyro-aided 2-point RANSAC.
20. Kneip, Scaramuzza & Siegwart, "A Novel Parametrization of the Perspective-Three-Point Problem," CVPR 2011. P3P on bearings for loop verification.

**Back end, loops and GNSS (Stages 4 and 5)**

21. Kaess, Johannsson, Roberts, Ila, Leonard & Dellaert, "iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree," IJRR 2012.
22. Yang, Antonante, Tzoumas & Carlone, "Graduated Non-Convexity for Robust Spatial Perception," RA-L 2020. Robust loop factors.
23. Lee, Eckenhoff, Geneva & Huang, "Intermittent GPS-aided VIO: Online Initialization and Calibration," ICRA 2020. ENU alignment and partial GNSS.
