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
| 2026-10-10 | 1a | Static initialization, `static_init_eval`, EuRoC IMU reader; gate scripts (`scripts/`, `python/msckf_gates/`) | 08ffd04 |
| 2026-10-11 | 1 | IMU preintegration (GTSAM cross-check), minimal KLT tracker and its evaluation; Basalt flow adopted (vendored `BasaltFlowTracker`, checked against Basalt); `tracker_compare`; benchmark runner (`scripts/run_benchmarks.sh`, `~/benchmark/<date-time>/`) | 2be8886 |
| 2026-10-11 | 1 | Stage 1 tracker gate on the vendored Basalt flow (FAIL: checkerboard aliasing on TartanAir, room1 > 2 px; deferred), TartanAir gross-error overlay | 7ca7e9d |
| 2026-10-11 | 1b | Dynamic initialization (sqrtVINS paper: feature-less linear solve, refinement with landmarks, keyframes, scale gate), retry pipeline with static and in-place starts; sim car and pushed cart, KITTI raw; `dynamic_init_eval`, `play_init.py`. Stage 1b gate PASS | 3bcae91 |
| 2026-10-11 | 1b | RMSE gate, no-refinement hand-off, triplet scale equations (off), stillness at the 90th percentile | c911ab7 |
| 2026-10-11 | 3 | MSCKF core (`src/filter/`): state, propagation with FEJ, cloning with t_d, feature update (nullspace, chi-square, QR, Joseph), sliding window; shared `geometry/triangulation`; `msckf_sim_eval`, `stage3_sim` gate. Handheld consistent; cart overconfident (see Stage 3 "Progress") | 77ba820 |
| 2026-10-12 | 3 | Review fixes: `stage3_sim` consistency on the share of epochs in the band (≥ 90 %), propagation refuses targets before the state time, nullspace projection in place, shared `maxRayAngleDeg` | 54698d0 |
| 2026-10-12 | 1b | Init review fixes: featureless init refuses < 6 equations, IMU must cover the window, linear hand-off covariance rederived, refinement covariance inverted once | 6d006eb |

### Handover (2026-10-12)

Read this, then the Stage 1b section, before continuing. The latest commit is 6d006eb; the working tree was clean at handover, apart from this plan update and the untracked `.clang-format` / `.clang-tidy`.

**Where things stand:**
- **Stage 0:** complete.
- **Stage 1a (static init):** gate PASS.
- **Stage 1b (dynamic init, sqrtVINS paper):** gate PASS on the retry pipeline, which tries static, then in place, then dynamic.
- **Stage 1 tracker gate:** FAIL, deferred by the user. It fails on repeating-pattern aliasing (checkerboard, AprilGrid) and on large motion without prediction.
- **Stage 3 equation audit (2026-10-17):** nine deviations from the plan artifact, listed in Stage 3 "Equation audit"; open ones are the error-state order, the missing ESKF reset Jacobian, non-FEJ extrinsic and bias Jacobians, and short tracks dropped at marginalization. Triangulation now uses anchored inverse depth (user decision, 2026-10-17).
- **Stage 1's full milestone:** the success criterion "VIO ATE over the next 10 s" needs the Stage 3 filter. Init results are judged on the hand-off state until then.
- **Benchmarks:** latest full run `~/benchmark/2026-10-12_004139` (commit 6d006eb, after both review-fix commits) against `2026-10-11_230242` (c911ab7): 0 regressions, 0 degraded. FAILs, all known: `stage1_tracker` (deferred), `basalt_flow_check` (0.156 / 0.131 % unmatched points vs 0.1 %, room1), `stage3_sim` (first run, now its baseline; same numbers as Stage 3 "Progress"). `tracker_compare` timings 3–23 % faster for every tracker, Basalt included: machine load, not code.

**The user's working rules** (also in `~/.claude/CLAUDE.md` and the repo's `CLAUDE.md`):
- **Git:** never `git commit` or `git push` (nor amend, rebase, tag or force-push). Stage, then propose a commit message ending with the Co-Authored-By line. The user commits and pushes.
- **Clean room:**
  - Never open GPL code (OpenVINS, VINS-Mono, ORB-SLAM3).
  - sqrtVINS is LGPL, and its code must never be read: **paper only** (arXiv 2510.10346; the HTML version is readable with a web fetch).
  - BSD code (Basalt, GTSAM, COLMAP, OKVIS2-X core) may be read for mechanics.
- **Python:** lint as you go with black, isort, flake8 `--max-line-length 120` and mypy (config in `python/pyproject.toml`). Run Python with `~/.venv/msckf/bin/python`.
- **Scripts:** analyses, downloads and gates are committed scripts in `python/` or `scripts/`; no throwaway scripts. Reuse existing code.
- **Display:** video with OpenCV imshow, images with matplotlib TkAgg or eog. Under WSL, set `DISPLAY=:0`. `python/viz.py`'s `FrameSink` handles windows (click to pause) and falls back to mp4 without a display.
- **Errors:** report and gate error metrics as **RMSE**, not medians; latencies such as time to init may stay medians.
- **Init goal:** sqrtVINS-style init; the user found it the most robust. On a cart driving a boustrophedon it "works the best out of any other option". Deviations from the paper are gaps to close or to justify.
- **Calibrations:** use published calibrations; no re-calibration side quests. Support only the camera models our data uses.
- **Plan:** keep this plan updated at every milestone, with decisions and measured numbers.

**How to run:**
- Build and test: `cmake --build build -j8`, then `cd build && ctest` (136 tests). There is no sudo; dependencies come through FetchContent.
- Gates: `scripts/run_gates.sh`, or a single gate as `python python/msckf_gates/stage1b_dynamic_init.py`, which writes `gate_out/<gate>/report.md`.
- Benchmarks: `scripts/run_benchmarks.sh` (see above).
- Init tools:
  - `build/dynamic_init_eval euroc|sim …` (options in its header comment: `--restart`, `--refine`, `--triplets`, `--sim-pixel-noise`, …).
  - `python python/play_init.py room1 [--cam cam1] [--t0 0]`: plays init attempts with gravity plumb lines and speed.
- Front-end tools:
  - `build/tracker_eval` (modes: tracks, tartanair, …; `--tracker basalt`);
  - `python/play_tracks.py` (TUM-VI, TartanAir gross errors);
  - `python/tracker_compare.py` (AprilGrid truth).
- Basalt pybind (used by `tracker_compare` and `basalt_flow_check`): the build in `~/work/basalt` (uncommitted pybind additions there), built static.

**Key code:**
- Initialization, in `include/msckf/init/` and `src/init/`:
  - `static_init`: Stage 1a.
  - `featureless_init`: the paper's linear solve, keyframes, least-squares covariance, and the optional triplet equations.
  - `init_refinement`: Gauss-Newton with landmarks and the joint covariance.
  - `dynamic_init`: the retry pipeline (static, in place, linear, refinement or linear hand-off, scale gate).
- Front end: `include/msckf/vision/basalt_flow.h` and `src/vision/basalt_flow.cpp`, the vendored Basalt flow (BSD notice).
- Simulator: `src/sim/`, with scenarios handheld, cart (boustrophedon with push surges and stops), fast_rotation and car.

**Priority (user, 2026-10-11):** after the init gates passed, go to Stage 3 (the filter). Prioritize handheld TUM-VI, EuRoC MAV and UZH-FPV; the cart is reported, not gated, until those work. Next for Stage 3, in order:
1. **FEJ observability check** (numerical rank, ‖O·N‖ on simulation): FEJ currently raises position NEES on fast rotation and the cart.
2. **MSCKF on real data:** a `msckf_run` tool on EuRoC / TUM-VI (vendored Basalt flow + Stage 1b init hand-off + filter), ATE / RE through `msckf_eval`; then UZH-FPV (the user is downloading it to `/mnt/c/Users/user/data/uzh_fpv`; a committed fetch/convert script is still needed).
3. **Zero-velocity update**, then the right-invariant variant (plan, Stage 3).
4. **Cart consistency** (deprioritized): see Stage 3 "Progress".

**Earlier next steps (Stage 2 and init), still open:**
1. **Stage 2, gyro-aided prediction** in the vendored flow. The user deferred it in Stage 1; it is now the proposed first Stage 2 item. Measure it with the Stage 1 tracker gate (TartanAir gross errors, room1) and, per the plan, on UZH-FPV (not downloaded yet).
2. **Stage 2, 2-point RANSAC on bearings with the gyro rotation (pipeline step 5).** The user declined bearing RANSAC earlier because the front end lacked the rotation; with gyro prediction it has it. This targets the aliasing tracks.
3. **Hybrid Shi-Tomasi/FAST detector**, for checkerboard X-junctions (deferred by the user).
4. **Zurich Urban MAV integration:** the facts and assessment are under Datasets. The user asked to add it, then paused the integration for this handover.
5. **Init items still open:**
   - cart scale tail (sim cart dynamic scale RMSE 28 %);
   - false in-place starts on decelerating turns (fix: a coherent-translation test);
   - EuRoC V1_01 overconfident;
   - the no-refinement covariance;
   - triplets to re-test on real cart data;
   - KITTI for diagnosis only.
6. **Real cart data**, with the sqrtVINS binary run as a baseline on the same segments. This is the real test of Stage 1 on the target platform.

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
| 1b dynamic init, retry pipeline (sim handheld, EuRoC MH_01–05, TUM-VI room1) | success ≥ 95 % within 5 s, time to init median ≤ 1 s, gravity RMSE < 1°, velocity RMSE < 0.1 m/s, sim velocity NEES in band | **PASS**: 100 / 100 / 100 %, 0.5 s, 0.45 / 0.43 / 0.45°, 0.041 / 0.084 / 0.042 m/s, NEES 3.00 |
| 1 tracker (`stage1_tracker`, vendored Basalt flow) | plan targets (room1 residuals > 2 px < 2 %, TartanAir gross < 2 %) | **FAIL** (deferred: repeating-pattern aliasing, no motion prediction) |

**How to reproduce.** Every gate result above comes from committed scripts, never ad-hoc commands:

- `scripts/data/fetch_tumvi.sh`, `scripts/data/fetch_euroc.sh` and `scripts/data/fetch_lamaria.sh` download, verify and extract the datasets.
- `scripts/run_gates.sh` builds, runs the unit tests and every gate checker (`python/msckf_gates/`, also installed as `gate-<name>` commands). Each gate writes its outputs and `gate_out/<gate>/report.md`, with a PASS/FAIL table and the ungated diagnostics.
- `scripts/run_benchmarks.sh` re-runs everything (unit tests, every gate, `tracker_compare`) into a fresh `~/benchmark/<YYYY-MM-DD_HHMMSS>/<benchmark>/` (root: `-r` or `$MSCKF_BENCHMARK_ROOT`), with a manifest (commit, dirty files) and the uncommitted diff, so every score is traceable to the code that produced it. Each benchmark writes `metrics.json` (value and better direction per metric; gates also their thresholds and pass/fail). `python/benchmark_compare.py` then compares with the previous run and writes `compare.md`: REGRESSION (a gate or check went from pass to fail), DEGRADED / IMPROVED (a metric moved by more than 2 % in its worse / better direction). The script exits non-zero on any failure or degradation.

Data lives in `~/data` ($MSCKF_DATA; overflow `/mnt/c/Users/user/data`, $MSCKF_DATA_LARGE), never in the repo:

- `~/data/tumvi/`: calib-cam1 and room1 (EuRoC format, 512×512), calib-cam8 (bag) and the published Kalibr results.
- `/mnt/c/Users/user/data/euroc/`: Machine Hall MH_01–05 (ASL format), from the ETH Research Collection bundle (record 20.500.11850/690084; the old robotics.ethz.ch links are dead). Also Vicon Room 1 V1_01–03, used only to tune noise (`python/tune_init_noise.py`), never by a gate.
- `~/data/tumvi/`: also calib-imu1–4 (AprilGrid, `tracker_compare`) and room2 (noise tuning only).
- `/mnt/c/Users/user/data/tartanair/`: TartanAir v2 trajectories (AmericanDiner, Office, OldTownFall, OldTownNight, RetroOffice), used by the tracker gate and `play_tracks.py`.
- `/mnt/c/Users/user/data/kitti/`:
  - `raw/`: unsynced "extract" drives 2011_09_26 0005, 0009, 0019, 0035, 0039, from `scripts/data/fetch_kitti_raw.sh`.
  - `euroc/`: converted by `python/kitti_raw_to_euroc.py`.
  - `flow2015/`: KITTI flow 2015.
  - Diagnosis only: see "Who does well on KITTI" in Stage 1b.
- `/mnt/c/Users/user/data/lamaria/training/`: LaMAria (ETH/Meta, ICCV 2025; Project Aria glasses, egocentric, city scale; data CC BY 4.0), the 23 training sequences (~170 GB) in the ASL export, from `scripts/data/fetch_lamaria.sh`; `python/lamaria_to_euroc.py` adds the `mav0/` layout (links to `aria/`, `sensor.yaml` from the pinhole JSON) and `gt_imu.csv`. Facts (2026-10-17):
  - Images are already undistorted to pinhole (~758×572, RGB PNG, 20 Hz, stereo). The IMU is Aria's right IMU at 1 kHz, **raw** (factory calibration not applied): the mean world specific force is 9.61 m/s² on R_01–R_07, 9.79–9.84 on R_08–R_13 and 9.65–9.77 on sequence_*.
  - Body frame = right IMU; the JSON's `T_b_s` is T_imu_cam; JSON and pGT quaternions are x, y, z, w.
  - Two kinds of pseudo ground truth: R_01–R_10 from MPS, world_from_imu at every frame; the control-point sequences (R_11–R_13, sequence_*) world_from_cam0 at keyframes 0.1–1 s apart (verified by gravity: read as world_from_imu, R_11's mean world specific force is (−0.8, −1.1, 1.8) m/s²; as world_from_cam0 all 11 such sequences give (0, 0, g) within 0.15 m/s²). Both worlds are z up. sequence_5_11/12 have control points only.
  - The 63 test sequences (~884 GB) have no ground truth (leaderboard only) and are not downloaded.
  - **Independent review of d79bf21 (2026-10-17):** frame and quaternion conventions, timestamps and noise units confirmed numerically (camera rotation from images vs ground truth: 0.8° median residual under the converter's convention, 12° under w, x, y, z). Fixed: MPS pGT repeats its first pose for 16–18 rows (~0.8 s, while the head turns) and holds a pose for a row or two mid-sequence; the converter now keeps only poses that differ from both neighbours. Re-scored (`~/benchmark/2026-10-17_175711`): retry success 39/44 (89 %), gravity RMSE 2.73°, velocity RMSE 0.293 m/s, scale RMSE 29 %, velocity NEES 6.03 (was 84 %, 3.41°, 0.332 m/s, 31 %, 4.50 on the frozen truth). `benchmark_compare` flagged a metric turning NaN as IMPROVED; NaN now counts as worse. `fetch_lamaria.sh`: files download via `.part`, only a 404 counts as "not on server", extraction leaves a `.extracted` marker, and a missing Content-Length skips the size check instead of failing.
  - Stage 1b reports the three short sequences R_01–R_03 "easy" (145–158 s each, MPS ground truth; user, 2026-10-17, replacing R_01 / R_04_medium / R_08_hard; all ten would add ~20 min of tracking per run) with zero bias and the EuRoC front end, and records their retry metrics (success, time to init, gravity and velocity RMSE) for `benchmark_compare`, ungated. R_01–R_03 (`~/benchmark/2026-10-17_152705`, f94d01c + review fixes): retry success 36/43 (84 %; failures: features 3, scale 2, diverged 1, pairs 1), time to init median 0.50 s (p90 1.25 s), gravity RMSE 3.41°, velocity RMSE 0.332 m/s, scale RMSE 31 %, velocity NEES 4.50; no window of any method is "accurate". First run, on the earlier R_01 / R_04 / R_08 set (`~/benchmark/2026-10-17_130606`, f94d01c): success 80/90 (89 %; failures: pairs 9, scale 1), time to init median 0.62 s, gravity RMSE 6.73°, velocity RMSE 0.746 m/s, window scale RMSE 81 %, velocity NEES 7.85. Far from EuRoC / TUM-VI (0.2–0.3°, 0.03 m/s). Suspects, untested: the uncalibrated IMU (2 % accelerometer scale on R_01–R_07), head-motion dynamics, the pGT's velocity truth (differenced 20 Hz poses).
  - Once `msckf_run` exists (Stage 3), the VIO metric is ATE against the MPS pGT (R_*) and the upstream control-point score and pose recall (`github.com/cvg/lamaria`, MIT code).
- `/mnt/c/Users/user/data/uzh_fpv/`: UZH-FPV drone racing (user download in progress, 2026-10-12); Stage 3 accuracy and fast rotation.
- `/mnt/c/Users/user/data/zurich_mav/`: the Zurich Urban MAV dataset (AGZ, Majdik et al., IJRR 2017). The user downloaded it with their own script, `zurich_mav/bash/download_and_extract.sh`; it is not yet integrated (see Datasets).

## Scope and rules

Build a monocular camera + IMU MSCKF in C++17, clean-room from papers only, in gated stages: initialization, front end, back end. No stage starts until the previous one passes its metric gate on both simulation and real data.

**Primary sources (math only, no code):**

- OpenVINS documentation and its ICRA 2020 paper (Geneva et al.): propagation, FEJ, MSCKF update, calibration Jacobians.
- Li & Mourikis, MSCKF 2.0 (IJRR 2013): closed-form transition matrix, observability, FEJ.
- Geneva & Huang, OpenVINS state-initialization report, and the sqrtVINS paper (Peng, Chen, Huang): static and dynamic initialization.
- Usenko et al., Double Sphere camera model (3DV 2018), and Kannala & Brandt (PAMI 2006) for camera models.
- Trawny & Roumeliotis, indirect Kalman filter report: reference for the JPL error-state algebra being translated.

**Clean-room rule:** GPL code (OpenVINS, VINS-Mono, ORB-SLAM3) is never opened while implementing. Permissively licensed libraries (Basalt, GTSAM, COLMAP, OpenSfM) may be read for mechanics: data structures, numerical tricks, test patterns.

Local reference checkouts in `~/work`, licenses checked: `gtsam` (simplified BSD; pinned at a74146dbe, built test-only by `scripts/deps/build_gtsam.sh`), `basalt` (BSD-3), `OKVIS2-X` (BSD-3 core; its `external/` and `supereight2` keep their own licenses, checked per file before reading), `kalibr` (BSD-3, except `ethz_apriltag2`, which is not read). Papers: Forster et al., on-manifold preintegration (arXiv 1512.02363v3).

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

**Hamilton reference:** Solà, "Quaternion kinematics for the error-state Kalman filter" (2017), uses exactly this convention (local-to-global, right perturbation, world-frame additive v and p errors). Use it line by line for the IMU and ESKF math, including the reset Jacobian (its Section 6.3). Use Mourikis and MSCKF 2.0 only for the multi-state parts (clones, nullspace projection, FEJ).

**Error-state order:** [δθ, δb_g, δv, δb_a, δp], as in the MSCKF papers, so F and Φ blocks line up with the paper. Solà orders [δp, δv, δθ, δa_b, δω_b], so permute when reading his matrices. (Not yet followed by the code: see Stage 3 "Equation audit".)

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

Implementation, step 1 (`include/msckf/init/featureless_init.h`, 2026-10-11), from the paper's Sec. V-A (Props. 1–2) only:

- Unknowns v and g of the first frame's IMU, in that frame; biases and extrinsics taken as known (as in the paper). For each keyframe pair, the translation direction t is the smallest-eigenvalue eigenvector of Σ nnᵀ over the common features' epipolar normals (bearings rotated into {I0} by the gyro); the preintegrated camera displacement s t = (ΔT_k − ΔT_j) v + ½(ΔT_k² − ΔT_j²) g + (α_k − α_j) + (R_k − R_j) p_IC is projected onto the two other eigenvectors, removing s. (The paper's eq. 60 writes the (v, g) block as −A; reading it as +A flips gravity, which the noise-free test catches.)
- Decisions the paper leaves open: all pairs of the given frames, pairs with < 10 common features skipped, normals not normalized (near-parallel bearings weigh little); the 6-unknown least-squares solution is followed by Gauss-Newton with |g| fixed (g on the sphere, like the paper's (α, β)).
- Excitation gate: smallest singular value of the stacked system with unit-norm columns ≥ 1e-3, and accelerometer excitation ≥ 0.2 m/s², measured as the spread of the specific force rotated into {I0} (gravity is constant there). Thresholds provisional, to be set on the real-data evaluation.
- Tests (`test_featureless_init`): exact without noise (gravity < 1e-3°, velocity < 1e-4 m/s); handheld with noise, 1 s windows, 50 starts over 5 seeds: all accepted, gravity error median 0.24° (p90 0.65°), velocity error median 0.032 m/s (p90 0.10 m/s); a constant-velocity window is refused (excitation 0.07 m/s²).

Evaluation (`tools/dynamic_init_eval`, gate `stage1b_dynamic_init`, 2026-10-11): the paper's protocol (a start every 10 s, windows 0.1–1.0 s, each window starting at a frame), bearings from the adopted front end, left camera only; EuRoC with its ground-truth biases as the paper's "prior calibration", TUM-VI with zero bias (its IMU data is calibrated); velocity truth from differentiated poses for both. A run is *accurate* if accepted and gravity < 1°, velocity < 0.1 m/s or 10 %, scale < 5 % (scale: least-squares ratio of the window's keyframe-pair translations to ground truth, stricter than the paper's Sim(3) fit over 10 s of VIO). First result, **linear solve only (no refinement): FAIL** at the 1 s gate.

| Data, 1 s window | Accepted | Accurate | Gravity / velocity / scale error (median, accepted) |
| --- | --- | --- | --- |
| Sim handheld (10 seeds) | 100 % | 75 % | 0.25° / 0.020 m/s / 1.7 % |
| EuRoC MH_01–05 | 96 % | 40 % | 0.30° / 0.040 m/s / 6.1 % |
| TUM-VI room1 | 100 % | 92 % | 0.34° / 0.033 m/s / 1.5 % |

- At 1 s the medians meet the plan's targets except EuRoC's scale (6.1 % vs 5 %); the shares fail mostly on scale.
- Short windows are far off (0.1–0.3 s: gravity 5–25°, scale errors of hundreds of %), as the paper says the linear solve alone is; its refinement step is what makes short windows work.
- The excitation gate is too weak: the singular-value test accepts most of those bad short-window solutions. To be revisited with the refinement (a relative conditioning measure, or the refined covariance).
- Cart (sim): rejected by the acceleration test in 98–100 % of runs, as intended (it starts on the static path). Fast rotation: gravity 1.25°, scale 48 % at 1 s; little translation for the direction t.
- Runtime: 0.2–3 ms per solve (plan: < 5 ms on Orin; measured on the workstation).

Step 2, refinement (`include/msckf/init/init_refinement.h`, 2026-10-11): one clone (R, p, v) per frame plus shared biases; every iteration relinearizes a prior on the first clone (loose tilt and velocity, tight yaw and position: the gauge) and on the biases, the IMU preintegration between consecutive clones as a residual, and the features (re-triangulated, null-space projected as in the MSCKF update, chi-square gated once with the first iterate's covariance, then fixed), and takes a Gauss-Newton step; covariance = inverse information at convergence. Decisions:

- IMU as a relinearized measurement, not a prior propagated once: the propagated-once version stalled at 0.17° from a 5° start without noise; this one converges to 1e-4°. This is the iterated EKF update in information form; the paper's square-root form is a numerical choice we do not need.
- Features are eliminated, not kept in the state (no SLAM features in v1); the hand-off is the IMU state at the last frame with its 15×15 covariance.
- Gate decision frozen after the first iterate, 20 iterations, convergence 1e-5: the first version oscillated (features swapping in and out) or stopped short.
- Simulation (`test_featureless_init`, biases drawn from the prior for a fair NEES): exact from a perturbed start without noise; 1 s windows gravity error median 0.60° (linear 0.88°, which cannot estimate the bias), NEES 7.82 in [7.24, 8.80]; 0.5 s: 0.67° (linear 2.09°), NEES 7.81, all converge. **0.3 s mono is not reliable**: a quarter of the runs converge to wrong solutions (2–12°) that fit as well as right ones and report ~1° of tilt sigma, so neither the cost nor the covariance can reject them.
- Runtime is **open**: 180–390 ms per 1 s window on the workstation (dense ~200-state solves), against the plan's 50 ms on Orin. To be optimized (block-sparse structure) before Stage 3.

**OKVIS2-X baseline** (read for mechanics, BSD): it has no dynamic initializer: attitude from the mean accelerometer reading (`ImuError::initPose`, i.e. assumes rest), velocity 0 under a speed prior, and its bundle adjustment corrects the start afterwards. ROVIO is similar (attitude from the accelerometer, velocity 0, features at a default inverse depth with large uncertainty, the iterated EKF converges with motion). `dynamic_init_eval` runs this start as method `okvis`: on moving starts its gravity error is 6.4° (sim), 1.9° (EuRoC), 16° (TUM-VI room1) and its velocity error is the full speed, which a filter cannot absorb the way a bundle adjustment can.

**Gate realigned with the plan (2026-10-11).** The first gate combined gravity, velocity and *window* scale into one per-run "accurate" share. The plan defines success as conditions 1–2 (well-posed and through the excitation gate; refinement converges with a valid covariance; condition 3, VIO ATE, in Stage 3) and gravity, velocity and scale as separate metrics, scale being a Sim(3) fit over 10 s of VIO (Stage 3). The gate now checks, for the refined pipeline at 1 s: success rate ≥ 95 %, gravity median < 1°, velocity median < 0.1 m/s; window scale and the all-targets share are reported.

| Refined, 1 s window | Success | Gravity / velocity (median) | Window scale (median) | All targets met |
| --- | --- | --- | --- | --- |
| Sim handheld | 100 % | 0.19° / 0.008 m/s | 0.7 % | 100 % |
| EuRoC MH_01–05 | **89.6 % (FAIL)** | 0.28° / 0.035 m/s | 4.9 % | 46 % |
| TUM-VI room1 | 100 % | 0.27° / 0.014 m/s | 0.3 % | 100 % (also at 0.5 s) |

**Keyframes as in the paper (2026-10-11; goal restated by the user: "sqrtVINS style initialization. It is the most robust algorithm I tested").** The paper selects keyframes "based on the average parallax", 3 for a 0.1 s window and 5 for 0.5 s; we had used every frame. `selectKeyframes` now picks at most 5, first and last always, evenly spaced in cumulative mean rotation-compensated parallax, for both steps. Effect: refinement runtime 180–390 ms → 9–16 ms per window (workstation); simulation 1 s gravity median 0.56° (all frames: 0.60°), NEES 8.24 in band; gate at 1 s: sim 100 % / TUM-VI 100 % / **EuRoC 91.0 %** success (all frames: 89.6 %), gravity 0.19° / 0.28° / 0.23°, velocity 0.015 / 0.015 / 0.030 m/s. EuRoC's 6 failures: excitation gate 3, too few usable features for the refinement 3 (near-hover windows, little parallax). For reference, the paper's own mono success on EuRoC (with the VIO condition, 0.5 m) is 79.6 % at 0.1 s, 92.0 % at 0.5 s, 96.5–100 % at 0.75–1 s; on TUM-VI 93.8–100 %.

**Features in the state, for initialization only (decided by the user, 2026-10-11: "keep features in the state like sqrtVINS - but only for initialization").** The refinement now estimates the landmarks jointly with the keyframe clones (global positions, triangulated once, then Gauss-Newton; eliminated per iteration by the Schur complement on their 3×3 blocks) and hands them to the filter with the IMU state and their joint covariance. The filter uses these landmarks only until they are lost and never adds others (Stage 3); the v1 rule of no SLAM features otherwise stands. Implementation decisions: two-phase chi-square gate (iterate with all landmarks until the step is < 1e-3, gate once there, converge to 1e-5 with the inliers; gating after one step from a 5° start rejected every feature); landmark convergence relative to distance. Results: noise-free from a 5° / 0.37 m/s start, state exact and 131 landmarks within 2e-5 m; Monte Carlo IMU NEES 8.24 and landmark NEES 3.00, both in their bands at 1 s (0.5 s: 7.98 / 2.59, in band); at 0.3 s most of the wrong solutions now fail to converge and are refused. Gate at 1 s: sim 100 %, TUM-VI 100 %, EuRoC 89.6 % (one more non-converging start than without landmarks). Runtime 15–38 ms per window on the workstation after removing two costs (joint-covariance products recomputed per landmark pair; `integrateRange` scanning the whole IMU log, now a binary search); the 50 ms Orin budget still needs measuring, and the dense per-landmark Jacobians are the next saving.

**Retry pipeline, car scenario and the scale gate (2026-10-11).** `initializeDynamic` (`include/msckf/init/dynamic_init.h`): keyframes → linear → refinement, the window growing from 0.5 s by 0.25 s up to 5 s on refusal. Simulated KITTI-like `car` scenario (rolling start, 8–12 m/s, braking to 5 m/s for 90° turns, 10 Hz camera, 100 Hz IMU, street-corridor landmarks: `SimConfig::corridor_*`). The user reports sqrtVINS initializes their real cart well (OKVIS2-X only sometimes), so cart and car starts must initialize dynamically, not only at stops. Findings and decisions:

- Priors on the first clone's tilt and velocity were centred on the linear solution (same data): with 1 m/s the car came out 3–5× overconfident. Now uninformative (0.5 rad, 10 m/s); handheld NEES unchanged.
- The paper has no excitation gate; ours (0.2 m/s² specific-force spread) refused most car and cart windows outright. Removed from the pipeline, replaced by a **scale gate** on the refined hand-off: velocity sigma ≤ 30 % of the estimated speed. Mono scale collapses in weakly accelerated windows (scene and trajectory shrink together and still fit) and the absolute covariance at a collapsed scale looks confident; the relative one does not. Sim: below 30 % consistent (velocity NEES 2.0–2.9), above mostly collapsed (scale off > 20 % in 52 of 73). Real data: solutions above 30 % were still accurate, and 10 % refused good ones (EuRoC 61 %). No absolute floor: it admitted collapsed solutions that look stationary; stationary starts belong to static init.
- Real-data consistency: on EuRoC/TUM-VI the confident solutions (σ_v < 10 % of speed) have velocity NEES 8.7 (3 dof): the noise model (datasheet IMU noise, 1 px bearings) is optimistic for real data, and the differentiated-mocap velocity truth adds noise. **Open**: inflate IMU and pixel noise, tuned on sequences outside the gate's set.

| Retry pipeline (gate run, reported) | Success | Time to init (median) | Gravity / velocity error (median, accepted) | Velocity NEES (3 dof) |
| --- | --- | --- | --- | --- |
| Sim handheld | 100 % | 0.50 s | 0.32° / 0.024 m/s | 3.00 |
| Sim car | 80 % | 2.12 s | 0.47° / 0.54 m/s (7 % of speed) | 3.72 |
| Sim cart | 40 % | 3.00 s | 0.12° / 0.035 m/s | **26.2** |
| Sim fast rotation | 84 % | 0.75 s | 0.32° / 0.038 m/s | 1.96 |
| EuRoC MH_01–05 | 94 % | 0.50 s | 0.27° / 0.033 m/s | 4.95 |
| TUM-VI room1 | 92 % | 0.50 s | 0.30° / 0.033 m/s | 7.44 |

Fixed 1 s windows (gated): sim handheld 98.6 %, EuRoC 89.6 %, TUM-VI 92.3 %: FAIL as before. The simulated cart cruises its 20 m straights at exactly 1 m/s (only vibration), where mono scale is unobservable: refinement fails to converge or keeps too few features, and the few accepted cart solutions are overconfident. Whether a real pushed cart (speed surges from pushing) is closer to this or to the car is open until cart data exists.

**Cart push, real-data noise, KITTI (user, 2026-10-11: yes to noise inflation and KITTI, add push variation to the cart).**

- Simulated cart now pushed: on moving legs the speed surges by ±0.15 m/s at 0.6 Hz (scaled with speed, stops stay exact; peak 0.57 m/s²). Retry pipeline on 10 seeds: **64/70 (91 %) initialize, median 1.25 s**, velocity error 0.11 m/s; velocity NEES 5.5 (6 above chi2_0.99): slightly overconfident, open.
- Real-data noise inflation is tuned by `python/tune_init_noise.py` on sequences no gate uses (EuRoC V1_01–03, TUM-VI room2; `fetch_euroc.sh vicon_room1`, `fetch_tumvi.sh room2`): IMU noise scale × pixel sigma grid, least inflation whose mean velocity NEES is in its 95 % band. `dynamic_init_eval --imu-noise-scale --pixel-sigma`.
- KITTI raw: `scripts/data/fetch_kitti_raw.sh` (unsynced "extract" drives, OXTS IMU at 100 Hz; default 2011_09_26 drives 0005, 0009, 0019, 0035, 0039, ~9.8 GB, CC BY-NC-SA, evaluation only) and `python/kitti_raw_to_euroc.py` (EuRoC layout: image_00, K_00/D_00 with k3, T_BS from calib_imu_to_velo · calib_velo_to_cam, OXTS rates and accelerations, pose ground truth from lat/lon/alt by the devkit's Mercator projection and roll/pitch/yaw). Checked on drive 0035: the world-frame specific force averages (0.28, −0.45, 9.81) m/s² and the gyro yaw rate matches the ground-truth heading to 0.005 rad/s. The Stage 1b gate reports every converted drive (a start every 5 s). First look (drive 0035 only, 14 s): accepted solutions inaccurate (velocity errors 0.4–5.3 m/s at ~6 m/s) and inconsistent; suspects: the front end at 10 Hz without motion prediction, camera–IMU time offset of the unsynced data, unknown OXTS noise.

**Noise tuning and KITTI results (2026-10-11).**

- `tune_init_noise.py` grid (IMU ×1–8, 1–3 px) on EuRoC V1_01–03 + TUM-VI room2: velocity error unchanged by inflation (~0.037 m/s), NEES falls 27.9 → 3.7; at **IMU ×4, 2 px** V1_02 (4.46), V1_03 (3.26) and room2 (3.48) are in or near band; V1_01 (slow, 0.2–0.4 m/s) stays ~3× overconfident at every setting (median NEES 11.8): open. Velocity truth checked: differencing EuRoC's smoothed positions over ±10 ms adds only 0.004 m/s. Adopted in the Stage 1b gate for EuRoC and TUM-VI (`REAL_NOISE`); simulation not inflated. Result: EuRoC retry NEES 1.23, TUM-VI 2.62.
- KITTI data issues found and fixed: OXTS timestamps jump back by up to 40 ms and have gaps up to 0.55 s (converter keeps a strictly increasing stream; the pipeline refuses windows with an IMU gap > 50 ms, `max_imu_gap`); velocity truth from differenced OXTS positions disagrees with the OXTS's fused velocity by 0.2 m/s median (0.5–0.7 m/s p90) at any baseline, so the converter writes the fused velocity into EuRoC's state columns and `dynamic_init_eval` uses state velocity when present. OXTS noise measured against ground truth (`python/kitti_imu_consistency.py`): gyro 1.74e-3 rad/s/√Hz, accelerometer 1.22e-2 m/s²/√Hz (10× / 6× ADIS), used by the converter; no further IMU inflation for KITTI. Camera–IMU time offset checked with `tracker_eval euroc --time-offset`: track residuals are lowest at 0 to −0.02 s (median 1.07 px, p90 4.0 px).
- KITTI retry (5 drives, a start every 5 s): 17/26 (65 %) initialize, median 2.5 s, gravity 0.93°, velocity error 0.89 m/s (speeds 4–14 m/s), velocity NEES 25 (from 971 before the fixes): still ~3× overconfident in sigma. Suspects: the front end at 10 Hz without motion prediction (heavy-tailed residuals, p90 4 px against a 2 px model), remaining OXTS model error.

**Stage 1b re-gated on the retry pipeline (user, 2026-10-11).** The gate now judges `initializeDynamic` as the system runs it, not one fixed window: per gated dataset (sim handheld, EuRoC MH_01–05, TUM-VI room1), success within 5 s ≥ 95 %, time to initialize median ≤ 1 s (plan: ≤ 1 s handheld), gravity median < 1°, velocity median < 0.1 m/s, and on simulation the mean velocity NEES in its 95 % band. Car, cart, fast rotation and KITTI reported; fixed windows reported as diagnostics. Result: **FAIL** on success rate only, everything else passes. (Superseded: **PASS** once the pipeline also takes static and in-place starts, below.)

| Retry pipeline | Success | Time to init (median) | Gravity (= roll/pitch) error median / p90 | Speed error median / p90 (relative median) | Velocity NEES |
| --- | --- | --- | --- | --- | --- |
| Sim handheld | 100 % | 0.50 s | 0.31° / 0.67° | 0.024 / 0.043 m/s (3.7 %) | 3.00 (band 2.45–3.60) |
| EuRoC MH_01–05 | **92.5 %** | 0.50 s | 0.25° / 0.74° | 0.030 / 0.116 m/s (6.2 %) | 1.23 |
| TUM-VI room1 | **91.7 %** (11/12) | 0.50 s | 0.31° / 0.46° | 0.035 / 0.056 m/s (3.7 %) | 2.62 |
| Sim car | 80 % | 2.1 s | 0.46° / 1.86° | 0.54 / 2.15 m/s (6.8 %) | 3.72 |
| Sim cart (pushed) | 91 % | 1.25 s | 0.20° / 0.48° | 0.069 / 0.275 m/s (7.9 %) | 5.50 |
| KITTI raw | 65 % | 2.5 s | 0.93° / 1.92° | 0.89 / 2.41 m/s (11.1 %) | 25.1 |

EuRoC's 5 failures: 4 refinement non-convergence, 1 scale gate; TUM-VI's 1: too few usable features.

**Static and in-place starts (user, 2026-10-11).** The TUM-VI room1 failure (start 0, both cameras) was not a dynamic-init failure.

- **What happens:** the camera is hand-held "still" for the first 6 s.
  - It rotates at 1–6°/s, with 0.1–0.4 m/s² of tremor and 2–8 px of raw disparity per 0.5 s, but translates only 1–3 cm/s.
  - That fails Stage 1a's stillness tests (accelerometer std 0.1 m/s², raw disparity 1 px).
  - The refinement refuses with "features": without translation nothing has the 0.2° parallax.
  - The linear solution was already within 0.5° and 0.15 m/s there.
- **The principle (user):** at rest gravity is the bias-corrected accelerometer reading, no speed needs estimating, and no odometry is needed, since the device is in place.
- **The change:** each attempt of `initializeDynamic` now tries, in order:
  1. **Static** (Stage 1a; disparity from the bearings via `pixels_per_radian`): truly still. b_g is the mean gyro.
  2. **In place** (`inPlaceInitialize`): rotating but not translating.
     - Gravity is the gyro-compensated mean specific force over the window (preintegrated: dRᵀ dv / T), with v = 0.
     - Velocity σ is 0.05 m/s (the residual translation of a hand), and enters the tilt covariance as (v_e − v_s)/T. b_g comes from the prior.
     - Accepted when the rotation-compensated median disparity (first-frame bearings rotated by the gyro into the last frame) is ≤ 1 px over ≥ 10 common features, and the mean specific force is within 0.3 m/s² of g.
  3. **Dynamic** (steps 1–2 and the scale gate).
- **Recording:** runs.csv's `init` column records the path. Unit test `InPlaceInit.RotationWithoutTranslation` checks that pure rotation at up to 0.3 rad/s gives tilt < 1e-3 rad, and that the same rotation at 0.3 m/s is refused (disparity > 5 px).

| Retry pipeline | Before | After |
| --- | --- | --- |
| TUM-VI room1, cam0 | 11/12 | 12/12 (start 0 in place at 2 s, gravity error 0.22°) |
| TUM-VI room1, cam1 | 11/12 | 12/12 (start 0 in place at 0.5 s, 0.42°) |
| EuRoC MH_01–05 | 62/67 | 67/67: the former failures were near-still starts (rotor vibration fails Stage 1a's accelerometer test); 6 starts in place, 1 static, gravity 0.05–0.46° |

Stage 1b gate: **PASS** (room1 success 100 %, median gravity 0.30°, velocity 0.028 m/s, NEES 2.40; EuRoC 100 %, 0.24°, 0.029 m/s, NEES 0.99). Sim, car, cart and KITTI are unchanged (no still starts). In-place NEES is ≈ 0 on EuRoC (v truly ~0), so 0.05 m/s is conservative there.

- **Restart on failure** (`dynamic_init_eval --restart S`, "restart" rows): before this change, restarting every 10 s initialized room1 start 0 at the 10 s restart (10.5 s).
- **Viewer:** `python/play_init.py room1 [--cam cam1] [--t0 0]`. It plays every attempt from a start, drawing the estimated (yellow) and true (green) gravity as plumb lines from the principal point, with the estimated vs true speed, from `trace.csv`.

How the references treat rotation-only starts (from their docs and papers, not verified in code):
- **Basalt:** gravity from the first accelerometer sample, v = 0, then the optimizer. Low-parallax points are simply not triangulated.
- **OKVIS2-X:** mean accelerometer reading plus a zero-speed prior. Its homogeneous landmarks keep points at infinity useful under pure rotation.
- **sqrtVINS (paper):** static or dynamic, with no explicit in-place case.

In-place init is a batch complementary filter, made safe by the vision "not translating" check. A feature-fitted homography is ambiguous with planar scenes, whereas the gyro-predicted one is not. It is still a candidate cross-check of gyro bias and time offset.

**Gate errors as RMSE (user, 2026-10-11: "rmse is better than median").** The Stage 1b gate checks gravity RMSE < 1° and velocity RMSE < 0.1 m/s (time to initialize stays a median, as a latency), and reports scale RMSE over dynamic hand-offs. Still **PASS**:

| Retry pipeline, RMSE | Gravity | Velocity (relative) | Scale |
| --- | --- | --- | --- |
| Sim handheld | 0.45° | 0.041 m/s (7.2 %) | 5.1 % |
| EuRoC MH_01–05 | 0.43° | 0.084 m/s (10.2 %) | 9.1 % |
| TUM-VI room1 | 0.45° | 0.042 m/s (3.9 %) | 2.9 % |
| Sim fast rotation | 0.55° | 0.064 m/s (20.8 %) | 12.4 % |
| Sim cart (pushed) | 0.29° | 0.157 m/s (18.4 %) | 25.0 % |
| Sim car | 1.35° | 1.39 m/s (18.4 %) | 23.7 % |

Scale against the milestone's < 5 %: met only on room1. The tails (worst 28 % EuRoC, 127 % cart, 103 % car) dominate the RMSE; not gated yet.

**The paper's no-refinement variant, and a limit of the feature-less solve (2026-10-11).** The paper (Sec. V-B) says only that "it is possible to skip iterative refinement and use a conservative initial covariance" (0.6 ms on a laptop). It gives no values, and it scores success by the ATE of the following 10 s of VIO, so a rough start that the filter later corrects still counts.

Implemented:
- `FeaturelessInitResult::covariance`: least-squares covariance of (v, gravity tangent). The residual variance comes from the fit, floored by the preintegrated position noise.
- `linearHandoff`: the linear solution propagated to the last keyframe, with that covariance mapped to tilt, position and velocity. Preintegration noise and bias priors are added.
- `InitRefinementPolicy` {kAlways (default), kFallback, kNever}.
- Eval: `dynamic_init_eval --refine`, `--linear-cov-scale`, and an `nr` row per fixed window.

Measured on the tuning sequences and simulation (`nr` rows, scale 1):

| NR, linear only | 0.1 s | 0.3 s | 0.5 s | 1 s |
| --- | --- | --- | --- | --- |
| Sim handheld: gravity RMSE / velocity NEES | 27° / 57 000 | 10.6° / 107 | 1.9° / 7.0 | 0.37° / 9.6 |
| EuRoC V1_01–03 | 4.4° / 11 (1 accepted) | 3.4° / 50 | 3.1° / 27 | 1.1° / 16 |
| TUM-VI room2 | 3.7° / 327 | 1.6° / 18 | 1.1° / 5.9 | 0.79° / 10.5 |
| Sim cart | 8.0° / 8 900 | 5.3° / 2 200 | 2.9° / 18 | 3.5° / 40 |
| Sim car | – | 2.2° / 48 000 | 1.6° / 26 000 | 4.4° / 58 |

The least-squares covariance is overconfident by 3× to 10 000×. Its errors are systematic: a wrong translation direction biases every equation alike, and the residuals stay small. So no fixed factor makes it conservative. kAlways stays the default, and NR remains a diagnostic until Stage 3 can score it by the paper's VIO criterion.

**Straight-path degeneracy of the feature-less solve.**
- **The cause:** each pair's equations are projected onto the plane orthogonal to that pair's translation direction, which removes its scale (Prop. 2). On a straight path every pair's translation is parallel, so the speed along the path vanishes from the system. The least-squares solution returns ≈ 0 for it, which is not an estimate.
- **The evidence:** the linear velocity error is 80–100 % of the speed on the sim cart and car at every window length (0.1–1 s). On handheld motion, whose path curves, it falls to 7 % at 1 s.
- **Who recovers it:** only the refinement does (cart 91 % success, 25 % scale RMSE), through landmark depths shared across frames. The 6 diverged cart refinements start from this ≈ 0 speed.
- **Consistent with the user's experience** (2026-10-11: sqrtVINS "works the best" on a cart driving a boustrophedon): the U-turns change the travel direction, which makes speed observable.
- **Candidate fix (a deviation from the paper, to justify on cart data):** add relative-scale equations from features seen in three or more keyframes. A common feature fixes the ratio of the pair scales s_ij / s_ik, which restores the along-track speed whenever the speed changes. Alternatively, give the refinement an along-track speed start from those triplets.

**Triplet relative-scale equations, tried (2026-10-11).** Implemented in `featurelessInit` (`FeaturelessInitConfig::triplet_scale`). For keyframes i, j, k, a common feature's unit-baseline depths at i give s_ij μ_ij = s_ik μ_ik. With ρ the median ratio, that adds one equation (t_ijᵀ(A_ij x + rb_ij) − ρ t_ikᵀ(A_ik x + rb_ik)) / √(1+ρ²) = 0. Translation signs come from positive depths, and features need ≥ 2° of parallax.

| Sim cart, 1 s, linear solve only | Without | With |
| --- | --- | --- |
| No pixel noise: gravity / speed error | 3.24° / 50 % | 0.19° / 7 % |
| 0.3 px noise | 2.66° / 53 % | 0.36° / 18 % |
| 1 px noise | 2.35° / 85 % | 1.08° / 63 % |

So the equations are right, and the simulator's 1 px noise swamps them: it is drawn independently per frame, where real tracking jitter is ~0.1–0.3 px and correlated.

Through the full retry pipeline they do not pay off. The refinement already recovers the speed from its landmarks, and the extra equations sometimes hand it a worse start that it accepts confidently. Sim cart: scale RMSE 28 → 119 %, NEES 5.2 → 93 (one 0.18 vs 0.90 m/s solution). Car: success 80 → 91 %, but scale 24 → 43 % and NEES 3.7 → 6.7. Handheld and fast rotation are unchanged. **Default off**; re-test on real cart data (`dynamic_init_eval --triplets`). Weighting the triplets by their ratio spread changed nothing and was removed.

**Stillness tests and turns (user, 2026-10-11).**
- **Simulation allowance:** the simulator's independent 1 px per-frame noise kept every still start from passing the 1 px disparity limit. `dynamic_init_eval` sim now adds 3σ (`still_noise_px`; 0 on real data). The sim cart's stops now start static: 11 starts at 0.01–0.16°. Cart success is 100 % and its NEES 3.96.
- **Quantile, not median:** stillness is tested at the 90th percentile of feature disparity (`still_disparity_quantile`). Outdoors most features are distant and barely move while the device translates, and any near feature that moves proves translation.
- **Remaining false in-place starts:** 8 on the sim cart, all at the end of a U-turn arc as the cart decelerates into a stop (0.1–0.6 m/s at window start). Gravity errors are 1–2° (NEES up to 13).
- **Why (user):** a turn is radial, not pure rotation, and the camera looks ahead. Translation along the optical axis barely moves features near the focus of expansion, and the sim's 4 px allowance hides ~4 cm of travel. On real data the limit is 1 px.
- **Why the IMU cannot help:** on an arc the accelerometer sees the centripetal acceleration v·ω (and any braking) as well as gravity. A static or in-place solution, which assumes zero acceleration, therefore tilts its gravity toward the outside of the turn: 0.73° at 0.5 m/s on the 2 m radius, 2.9° at 1 m/s. The gyro-compensated specific force cannot reveal it (constant braking is indistinguishable from tilt), so only vision can.
- **The dynamic path is unaffected:** it models the motion through preintegration (sim cart gravity RMSE 0.32° on dynamic starts).
- **Next step if real cart data shows it:** a coherent-translation test, i.e. fit the expansion pattern (epipole) to all rotation-compensated flow. Its noise falls as 1/√N, rather than thresholding a per-feature quantile.

**Init code review (2026-10-12).** Fixes: the linear solve refuses fewer than six equations (two pairs, no triplets read a nonexistent 6th singular value: on sim fast rotation 5 "conditioning" refusals were that garbage, now "pairs"); the retry refuses a window the IMU does not cover from its start ("imu", was an uncaught throw) and finds a gap straddling the window start; static/in-place successes clear the failed attempts' linear fields; growth must be > 0. The **linear (NR) hand-off covariance** was rederived: the position row had a gravity term that cancels and lacked the tilt coupling; the preintegration dtheta was rotated although already in the last frame; the bias priors were not propagated. They now enter through the window's deltas and through the linear solution (re-solved with the bias perturbed, central differences), with the bias cross-covariance kept. `test_dynamic_init`: tilt + velocity NEES 4.57 (5 dof, 100 windows, biases drawn from the prior), band [4.40, 5.64]. Stage 1b gate (`python -m msckf_gates.stage1b_dynamic_init`): still **PASS**, retry path unchanged (refined hand-off). NR rows (not gated) accept less, since the bias prior now widens the velocity sigma: 1 s windows sim handheld 100 → 80 %, EuRoC 82 → 51 %, TUM-VI 92 → 77 %, fast rotation 40 → 13 %; on sim the biases are given exactly, so there it is conservative.

Remaining gaps to sqrtVINS: (2) the paper's no-refinement variant: implemented, but with no usable conservative covariance yet (above); (3) the window-retry policy of our step 3.

Earlier result, all frames: EuRoC: 7 of 67 starts fail (excitation gate 3, too few usable features 3, no convergence 1). Its window scale misses 5 % in half the runs, tracking excitation: EuRoC's 1 s windows carry 1.1–1.8 m/s² of acceleration against 2.4 (TUM-VI) and 3.5 (sim), and the scale failures drop from 9/15 in the lowest excitation quartile to 3/15 in the highest. Mono scale is observable only through acceleration; the plan's answer is to retry with a longer window (step 3), not yet implemented.

**Cart baseline.** sqrtVINS is the only initializer observed to work on the pushed cart, so it's the reference for cart data. Run it as an unmodified binary on the same segments and windows; it's LGPL-3.0, and its source stays closed under the clean-room rule (user: paper only). On cart data, the Stage 1 gate is matching its success rate.

### Stage 1 metrics

The protocol is adopted from sqrtVINS, so results compare directly with its published monocular numbers. Split each sequence into 10 s segments and, at each segment start, run initialization with windows of 0.1, 0.2, 0.3, 0.5, 0.75 and 1.0 s. A run succeeds only if all three hold:

1. The linear system is well-posed and passes the excitation gate.
2. Refinement converges and produces a valid covariance.
3. VIO position ATE over the following 10 s stays below a threshold, reported at 0.1, 0.3 and 0.5 m.

Run on simulation (50 seeds per scenario), EuRoC and TUM-VI (left camera only), and cart data. This judges initialization by whether the VIO that follows actually works, not just by the initial state. Targets below are proposed starting points.

Gate split (decided at the start of Stage 1): condition 3 needs the filter, which is Stage 3. So the Stage 1 gate covers everything measurable against ground truth at hand-off: conditions 1 and 2, gravity, velocity and scale error, initial NEES and runtime. The success rate with condition 3 (VIO ATE over the next 10 s) is re-run, and gated, once Stage 3 exists; the targets below stay as they are.

Build order: 1a static init (needs only the IMU, plus the tracker's median disparity when available), IMU preintegration (cross-checked against GTSAM), a minimal KLT tracker on our camera models that feeds bearings and disparity to the initializer (sqrtVINS style; the Stage 2 front end replaces it), then 1b: linear solve, refinement, excitation gate.

IMU preintegration (`include/msckf/imu/preintegration.h`), decided in implementation:

- Forster et al. (T-RO 2017): ΔR, Δv, Δp in the frame at t_i, bias Jacobians for the first-order bias correction, and the 9×9 covariance in the filter's order (δθ, δp, δv, all in frame i; GTSAM's covariance uses body-frame p, v errors and is rotated for comparison).
- Two step schemes. `kForster` rotates each step's acceleration with ΔR at the step start, as the paper and GTSAM do. That ignores the rotation within the step and is first order: 1.6 mm position error after 1 s of handheld motion at 400 Hz, 14 mm under fast rotation, comparable to the IMU noise over the same window. `kMidpoint` (default) uses ΔR Exp(ω dt / 2), with the half-step rotation carried into the error and bias Jacobians: second order, 10 µm and 78 µm. `kForster` is kept for the GTSAM cross-check.
- `integrateRange` averages the two samples bounding each interval, linearly interpolated at clipped ends.
- Tests: exact simulator states predicted (both schemes, convergence order 2 vs 1 measured), bias Jacobians vs finite differences (3e-7), covariance vs 500 Monte Carlo runs (NEES in band), and `kForster` against GTSAM's ManifoldPreintegration step for step (`test_preintegration_gtsam`, built when GTSAM is found).

Minimal KLT tracker (feeds bearings and disparity to the initializers; the Stage 2 front end replaces it): OpenCV pyramidal LK, forward-backward check, Shi-Tomasi grid detection masked by the camera model's valid region, and RANSAC on unit bearings with angular residuals. Evaluation, decided before implementation:

| Layer | Data | Ground truth | Measures |
| --- | --- | --- | --- |
| Exact multi-frame | TartanAir v2 (CC BY 4.0), left-front pinhole 640×640, 10 Hz: RetroOffice easy/hard, Office hard, OldTownFall easy/hard, OldTownNight hard, AmericanDiner hard | Depth + poses: each feature's true position in every later frame, with occlusion and depth-edge checks (corners often sit on depth edges, where interpolated depth is not trustworthy; such features are excluded and counted) | Pixel error vs track age (drift), accepted-but-wrong rate, RANSAC precision/recall on real failures |
| Real single step | KITTI Flow 2015 (CC BY-NC-SA, evaluation only), 200 training pairs | LiDAR flow, 10 → 11 | End-point error and outlier rate on real driving images |
| Real long tracks | EuRoC MH_01–05 and TUM-VI room1, cam0 and cam1 each as a mono sequence; later KITTI odometry and KITTI-360 fisheye | Ground-truth camera poses (mocap pose + extrinsics) | Triangulation residual of each track (≥ 5 observations), pairwise angular epipolar error, outlier rate, each above a noise floor measured on AprilGrid corners (TUM-VI calib-cam1) |
| Statistics | All of the above | none | Track length (median, survival), features per frame, coverage, runtime |
| Disparity for init | EuRoC still windows (static-init gate) | IMU stationarity + ground-truth motion | Separation of still and moving windows by median disparity |

A rendered ray-caster (exact truth for every camera model) was considered and dropped as overkill: TartanAir gives exact truth on realistic images, its toolkit can re-render Double Sphere and EUCM if needed, and the fisheye models are covered by the real-data layers. Datasets come from `scripts/data/fetch_tartanair.sh` and `fetch_kitti_flow.sh`.

Primary tracker goal (user, 2026-10-11): **track each feature as long as realistically possible, with few drop-and-rediscover events.** Measured on TartanAir, where depth and poses give each feature's true visible span (in the valid region, not occluded), followed past the track's death:

- **Completeness:** tracked length / visible length per track.
- **Premature loss:** ended tracks whose point was still visible, by cause (LK failure, left valid region, forward-backward, RANSAC). The tracker reports every dropped feature with its reason.
- **Re-detection rate:** new detections whose 3D point projects within 2 px of a dropped, still visible track's point (a duplicate landmark for the filter).
- **Overtracked:** tracks followed past their point's visibility (tracking the wrong thing).

First measurement (RetroOffice easy P000, default settings): median tracked length 1 frame against 7 visible, completeness median 0.25, 75 % of ended tracks premature (mostly the 0.5 px forward-backward check), 18 % of new tracks re-detections. Settings are tuned with `python/tracker_sweep.py`, which reports these metrics next to the accuracy guards (gross errors beyond the floor) for a grid of settings.

Findings that shaped the evaluation:

- **TartanAir's truth is not sub-pixel exact.** An oracle (each detection patch aligned straight into the later frame, starting at the true pixel, no tracker involved) disagrees with the depth/pose truth by a median of 0.45 px after one frame and 1.1 px after five, the same as the tracker. Pose timing, depth scale, focal length and principal point were checked and are not the cause; the rendered images and the geometry simply disagree at that level. TartanAir is therefore judged relative to the oracle floor, never in absolute pixels.
- **Real data with pose ground truth only** (EuRoC, TUM-VI): the triangulation residual mixes tracker, ground-truth and calibration error. On TUM-VI the floor is measured with AprilGrid corners (exact static points) on calib-cam1 through the same ground truth: median 0.38 px, p99 1.2 px, against KLT's 0.77 px and 9.4 px on the same sequence. EuRoC has no such board in the Machine Hall; its residuals are reported, not gated. A camera-to-ground-truth time offset was ruled out (0 ms is a sharp minimum).
- **KITTI Flow**: the ground truth is semi-dense, so sampling uses all four neighbours when valid and otherwise inverse-distance weighting within 1.5 px (requiring four valid neighbours left ~5 % of features evaluable).
- **Robust estimation:** LO-MSAC (MSAC scoring, local optimization by iterated reweighted least squares) replaces plain RANSAC: mean F1 0.973 vs 0.937 with 50 % outliers and noise near the threshold. MAGSAC++ is the next step if a gap remains.
- **Track length on real data is observation-weighted.** The baseline per-track median on EuRoC was 2–5 frames although only ~6 of 200 features are replaced per frame (a typical live feature lasts ~35 frames): the per-track median counts every weak detection that dies within a frame or two. The gate uses the median over observations of the length of the track each belongs to (how long the tracks feeding the filter are), and reports the per-track median with the share of tracks dying within 2 frames.
- **Optional drift layer:** MPI Sintel (exact dense flow and occlusion masks for every consecutive pair) would give an exact multi-frame drift measurement by chaining flow, with RANSAC off (non-rigid scenes). Not yet adopted.

Goal refined (user, 2026-10-11): **a high number of stable tracks.** Long tracks alone are not enough; the filter needs many of them, spread over the image, at every frame.

Tuning sweep (`tracker-sweep`, TartanAir P000 of the 7 sets; forward-backward 1 px, LO-MSAC 1 px unless noted):

| Setting | Completeness median / mean | Premature loss | Re-detected | Gross (> floor + 2 px) |
| --- | --- | --- | --- | --- |
| Defaults (3 levels, block 3) | 0.33 / 0.49 | 70 % | 27 % | 4.7 % |
| 4 pyramid levels (5: identical) | 0.43 / 0.51 | 67 % | 23 % | 3.7 % |
| + constant-velocity prediction | 0.42 / 0.51 | 68 % | 24 % | 3.4 % |
| + Shi-Tomasi block 7 | 0.50 / 0.54 | 64 % | 20 % | 3.8 % |
| + block 11 | 0.50 / 0.54 | 64 % | 17 % | 4.0 % |
| + block 11, quality 0.05 | 0.50 / 0.57 | 61 % | 16 % | 4.0 % |

A 31 px LK window was worse. Overtracking rises to ~4 % at quality 0.05. The plateau is the translation-only, raw-intensity, frame-to-frame LK model: ~60 % of ended tracks still die while their point is visible, 74 % of forward-backward drops happen on the first step after detection (median round trip 7.5 px), and gross errors stay ~4 % (gate: 2 %).

**Head-to-head against Basalt (decided 2026-10-11): if Basalt's tracker is better, use it** (BSD-3). Basalt's frame-to-frame flow differs on exactly the weak points above: an SE2 warp on a sparse 52-point patch, mean-normalized intensities (gain invariant), inverse-compositional Gauss-Newton, a 0.3 px forward-backward check, and one FAST corner per empty 50 px cell.

- **Method:** Basalt's own code runs unmodified, not a re-implementation. A hand adaptation was written and discarded in favour of this. The local Basalt checkout's pybind11 module (`src/python/basalt_pybind.cpp`, built with `BASALT_BUILD_PYTHON_BINDINGS=ON`) gains `basalt.OpticalFlow(model, params, width, height, config)`. It wraps `OpticalFlowFactory` (VioConfig defaults, any field overridable) through Basalt's own input and output queues; `process(t_ns, image)` returns the live keypoint ids and pixels. A committed script runs it over a sequence and writes the tracks. A replay mode in `tracker_eval` then scores them with exactly the evaluation used for our tracker. Basalt does not report why a track died, so its drop causes are "unknown". `tracker_eval` and `KltTracker` already share a `FeatureTracker` interface (`include/msckf/vision/feature_tracker.h`), which the replay plugs into.
- **Fairness:** both trackers are run at Basalt's defaults and at a matched feature budget (Basalt's 50 px grid gives ~144 cells on 640×640, against our 200 features; a 40 px grid gives ~256). Our tracker is run at defaults and at the best sweep setting. The same images and the same valid-region mask are used for both.
- **Status (2026-10-11):** the binding is built (`~/work/basalt`, build dir `build/relwithdebinfo` configured with `BASALT_BUILD_SHARED_LIB=OFF`, because the shared libbasalt cannot link vcpkg's non-PIC static ffmpeg) and tested (`test/python/test_basalt_pybind.py`: a known 2.3 / -1.7 px Fourier shift tracked to < 0.05 px RMS). It also exposes Basalt's AprilGrid detector (`detect_aprilgrid`, `aprilgrid_corners_3d`). Instead of a replay mode in `tracker_eval`, both trackers write the same `tracks.csv` (`tracker_eval tracks`, and the Basalt runner in `python/tracker_compare.py`) and one Python scorer computes every metric.
- **How Basalt detects:** only to spawn tracks. After tracking, FAST runs only in grid cells without a live point, one corner per empty cell (threshold 40 halved down to 5); live tracks are never re-detected and a lost track is never recovered. `frame_to_frame` aligns against the previous frame's patch, starting at the previous position (no motion prediction); Basalt's `patch` flow aligns against the patch from the detection frame (no accumulation) and is scored too.

Comparison metrics, chosen for the refined goal. Every one is reported per tracker, per sequence, on TartanAir (with truth) and on EuRoC / TUM-VI cam0 and cam1 (statistics only):

| Question | Metric | How |
| --- | --- | --- |
| How many stable tracks? | Stable tracks per frame | Median over frames of live features with age ≥ 5 and ≥ 10; on TartanAir only those still within floor + 2 px of truth |
| How well spread are the detections? | Cell coverage | Share of the valid region's cells (8×8 grid; a cell counts if ≥ 50 % valid) holding ≥ 1 live feature, median over frames; separately for all features and for stable (age ≥ 5) ones |
|  | Evenness | Normalized entropy of features per cell (1 = perfectly even), median over frames |
|  | Detection placement | Share of new detections landing in empty cells vs next to live features (≤ min. distance), the latter being wasted budget |
| How long do tracks survive? | Survival curve | Kaplan–Meier estimate of P(track age ≥ k), tracks alive at sequence end censored; reported: median survival and survival at 5, 10, 20 frames |
|  | Observation-weighted length | Median length of the track each observation belongs to |
|  | Completeness, premature loss, re-detection, overtracking | As defined above (TartanAir) |
| Are the tracks right? | Accuracy guard | Gross-error share and excess over the oracle floor (TartanAir); triangulation residual against the AprilGrid floor (TUM-VI) |
| Cost | Runtime | ms per frame, same machine, single thread for both |

Decision rule: adopt Basalt's tracker if it gives more stable tracks per frame and longer survival at no worse coverage, and passes the accuracy guards. If it wins on survival but loses on coverage, adopt its tracking and keep our grid detection. Either way the result and the chosen settings are recorded here.

**Drift truth on real images: TUM-VI calib-imu1–4** (static AprilGrid, 20 Hz, fast excitation motion; the calib-cam sequences are only 4 Hz). Basalt's detector gives the corners of every frame; with the published pinhole-equi calibration (Basalt's kb4) their bearings give each frame's board-to-bearing homography (DLT on unit bearings, corners > 1 px refitted out, frames with fit RMS > 0.5 px dropped; median fit 0.09 px). A feature inside the detected board is mapped to board coordinates in its first such frame and predicted in every later one; drift is the pixel distance to that prediction. The floor is the same scoring on the detected corners themselves. Two failure modes are separated:

- **Aliasing:** the grid repeats (pitch ~33 px in the image), and with no motion prediction a tracker whose search starts more than half a pitch from the truth converges on the same pattern on the neighbouring tag. Traced (Basalt track 2270, calib-imu1 frames 228→229): the board moved 19 px up, the track 13.7 px down, landing exactly one tag row off. Counted as a track whose board offset is within 0.2 pitch of a non-zero multiple of the pitch. Natural scenes rarely repeat like this, so aliasing is reported separately from gradual drift.
- **Grid corners:** FAST does not fire on the AprilGrid's X-junctions, Shi-Tomasi does (Basalt's nearest feature to a corner is ≥ 3.2 px away, ours median 0.3 px). Drift at corners is therefore only defined for our tracker; the board-wide homography truth covers both.

Results (`python/tracker_compare.py`, 2026-10-11; medians over sequences, cam0 and cam1 each a sequence; Basalt with its shipped TUM-VI / EuRoC config, also used for TartanAir):

| Data | Tracker | Coverage (stable) | Evenness | Survive ≥ 5 frames | Die within 2 | Drift > 1 px | Aliased | Error at age 5 / 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calib-imu1–4 | ours | 66 % (59 %) | 0.82 | 39 % | 53 % | 61 % | 7.5 % | 0.26 / 0.58 px |
| | Basalt | 80 % (63 %) | 0.90 | 25 % | 70 % | **2.1 %** | 22 % | **0.07 / 0.15 px** |
| | floor (corners) | | | | | 2.6 % | 0 | 0.08 / 0.10 px |
| room1 | ours | 80 % (50 %) | 0.89 | 35 % | 55 % | | | |
| | Basalt | 86 % (60 %) | 0.93 | 22 % | 71 % | | | |
| EuRoC MH_01–05 | ours | 78 % (70 %) | 0.90 | 42 % | 48 % | | | |
| | Basalt | **98 % (81 %)** | 0.96 | 21 % | 72 % | | | |
| TartanAir (7 × P000) | ours | 83 % (13 %) | 0.94 | 8 % | 81 % | | | |
| | Basalt | 95 % (0 %) | 0.96 | 0.6 % | 96 % | | | |

- **Spread:** Basalt wins everywhere (one FAST corner per empty cell). Our tuned setting (quality 0.05) loses a third of the coverage.
- **Accuracy:** Basalt's tracks barely drift: at the AprilGrid floor through age 20, and 2 % exceed 1 px against 61 % of ours. Its losses are aliasing jumps (22 %, a repeating-pattern effect) rather than drift.
- **Survival:** ours lives longer on every dataset; Basalt's 0.2 px forward-backward limit discards anything uncertain (70 % of its tracks die within 2 frames). On TartanAir (10 Hz; median motion 16–49 px per frame, p90 up to 85 px, against 3 px on EuRoC and 13 px on room1) Basalt keeps almost nothing: without motion prediction the search starts outside its convergence range. Gyro-aided prediction (deferred, user 2026-10-11: "this will come later when we want to squeeze more out of Basalt") targets exactly this and the aliasing.
- Runtime per frame: Basalt 4–9 ms, ours 9–14 ms (same machine, Basalt via its module with TBB).

**Decision (user, 2026-10-11): adopt Basalt's front end, then compare it with sqrtVINS's and try to improve on it.**

- **Adoption by vendoring:** Basalt's flow (frame-to-frame and patch variants, the patch alignment, grid FAST detection, the image pyramid) is copied into `src/vision/` under BSD-3 with attribution, behind `FeatureTracker`, without TBB or vcpkg. Gate: identical ids and pixels to the unmodified Basalt module on the same sequences.
- **sqrtVINS (LGPL-3.0): paper only.** Its code is never opened (user: "don't look at sqrtVINS code. Just the paper"); only the license file was checked. Its front end is re-implemented clean-room from the paper (Peng, Chen, Wu, Huang, T-RO 2025, arXiv 2510.10346) as another `FeatureTracker` and scored with the same metrics.
- **Vendored (2026-10-11):** `include/msckf/vision/basalt_flow.h`, `src/vision/basalt_flow.cpp` (`BasaltFlowTracker`; frame_to_frame and patch, patterns 24/50/51/52, optional valid-region mask). basalt-headers (image, pyramid; byte-identical to Basalt's) moved from test-only to a build dependency. `python/basalt_flow_check.py` gates it against the unmodified module on one sequence per dataset (500 frames): points are matched by position within each frame (not by id, since one point dropped in only one run shifts every later id), and at most 0.1 % of either run's points may lack a partner within 0.01 px. frame_to_frame (adopted) passes everywhere, ≤ 0.02 % unmatched; TartanAir exactly. Not bit-identical: float evaluation order differs (Basalt's -O3 -march=native alone does not reproduce it). Patch mode is reported, not gated: up to 0.6 % unmatched on EuRoC, whole points a median 14 px away, from frame 15 on; its fixed templates make one borderline forward-backward decision persist. Consistent with float sensitivity, unproven; to be resolved if patch mode is ever adopted.
- **Stage 1 tracker gate re-targeted (2026-10-11)** to the adopted front end: `stage1_tracker` runs the vendored flow (`tracker_eval --tracker basalt`, now accepted by every mode) with Basalt's shipped settings pinned in our repo (`msckf_gates.common.FRONT_END_TUMVI` / `FRONT_END_EUROC`: 40 / 50 px grid, 0.2 px forward-backward, pattern 51, 3 levels), plus the valid-region mask so every feature has a bearing. The gate no longer depends on the Basalt checkout. Its synthetic layer is `test_basalt_flow` (known sub-pixel shift: 0.035 px RMSE frame-to-frame, 0.030 px patch; < 2 % of drops away from the image edges; mask keeps features in the model's domain). Thresholds unchanged (the plan's targets). `--tracker klt` re-runs the original KLT gate. First result (`gate_out/stage1_tracker/report.md`): **FAIL**. Passes EuRoC track length (observation-weighted median 50–99 frames; KLT 2–5, failed), room1 median residual 0.10 px above the AprilGrid floor (KLT 0.21), TartanAir median error (at or below the oracle floor), synthetic and disparity. Fails room1 residuals > 2 px (3.4 / 3.3 %, target < 2 %; KLT 8 %) and TartanAir gross errors (2.4–25.9 %, target ≤ 2 %; KLT 1.3–12.9 %). The failures are a minority of grossly wrong tracks: on AmericanDiner the tracks alive at age 10 are a median 90 px off, locked onto repeating texture consistently enough to pass the forward-backward check. Causes: no motion prediction (TartanAir moves 16–49 px per frame; gyro prediction deferred) and no geometric outlier rejection (Basalt's mono front end has none; our pipeline step 5 does, and `FeatureTracker::rejectEpipolarOutliers` already implements it). Seen with `python/play_tracks.py AmericanDiner/Data_hard/P000` (gross errors drawn red, with a line to the true pixel): the gross errors sit on the checkerboard floor, each track jumped one tile to the matching corner and stayed locked (frame 134: 23 of 23 evaluated tracks gross; booths, tables and walls clean). The same repeating-pattern aliasing as on the AprilGrid. Bearing RANSAC in the front end (fits the relative motion to the tracks themselves, no IMU needed) was proposed and declined for now (user, 2026-10-11).

**Deferred front-end improvements** (user, 2026-10-11: note and move on; the stage 1 tracker gate stays at FAIL on TartanAir gross errors and room1 residuals > 2 px until one of these lands):

- **Hybrid Shi-Tomasi / FAST detector.** Checkerboard-like patterns (TartanAir AmericanDiner's floor, the AprilGrid) are where FAST is weak: its corners sit on the X-junctions, which repeat tile to tile, and the tracks jump to the identical neighbour (seen in `play_tracks.py AmericanDiner/Data_hard/P000`). Shi-Tomasi responds to X-junctions properly (on the AprilGrid ours sit 0.3 px from the corners, Basalt's FAST never closer than 3.2 px). Keep Basalt's one-per-empty-cell grid, score candidates with both.
- **Gyro-aided prediction** in the vendored flow: start each alignment at the rotated bearing, so motion beyond half a pattern period (TartanAir: 16–49 px per frame) no longer converges on the neighbour.
- **Bearing RANSAC** (declined for now): reject tracks inconsistent with the relative motion fitted to all tracks.
- `tracker_compare` variant `vendored` runs it, so the adopted tracker is in every benchmark run; `python/play_tracks.py` plays a TUM-VI recording with its tracks (imshow).
- **Improvements** are measured on top of the vendored Basalt flow. Gyro-aided prediction is deferred until we want to squeeze more out of Basalt (user, 2026-10-11).

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
| Track length | Observation-weighted median frames per track (see Stage 1 notes), and share reaching window length | Median ≥ 8 at 20 Hz on EuRoC |
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

1. **Triangulate** each finished track: anchored inverse depth in the first observing clone, Gauss-Newton on bearings. Reject on low parallax, poor conditioning, or rays pointing away from the point. (Implemented: ρ < 0 rejected, points at infinity (ρ = 0) kept; the filter linearizes in the same parameters: see "Progress".)
2. **Linearize** the pixel residual through the camera model's Jacobian. With ⁱp_f = R_GIᵀ(p_f − p_GI) and ᶜp_f = R_ICᵀ(ⁱp_f − p_IC), the clone blocks are ⌊ⁱp̂_f×⌋ for δθ, −R̂_GIᵀ for position, and R̂_GIᵀ for the feature, each multiplied by R_ICᵀ and the camera Jacobian.
3. **Nullspace projection** with Givens rotations, then a χ² gate at 95% with 2M − 3 degrees of freedom.
4. **Stack and compress** all accepted features with QR, then a Joseph-form update and injection R ← R̂ Exp(δθ).
5. **Marginalize** the oldest clone when the window is full, after using every track that touches it.

### Zero-velocity update

When the IMU and the front-end disparity both say stationary, apply v = 0 as a measurement and stop adding clones. Without it, a stopped cart lets velocity and bias errors grow, since the visual update has no parallax.

The same holds "in place" (user, 2026-10-11: no odometry while in place). A hand-held device rotating without translating should also hold position with v = 0. Use the test from Stage 1b's in-place start: rotation-compensated disparity ≤ 1 px, rather than the raw-disparity and accelerometer-spread tests, which tremor and rotation fail. Rotation keeps being integrated from the gyro.

### Error parameterization: FEJ vs right-invariant

The target is GNSS-denied or partially denied operation, so yaw and position stay unobservable for long stretches and consistency matters. Both error parameterizations are implemented behind the same interface: standard multiplicative error with FEJ, and right-invariant SE₂(3) error. Only Φ, the clone Jacobian, H and the reset differ; clones, nullspace projection and QR are shared. Keep whichever has the better NEES over long simulated runs; ATE decides only if both are consistent.

### Progress (2026-10-11)

Code: `include/msckf/filter/`, `src/filter/` (`filter_state`, `propagator`, `feature_update`, `msckf`, `sim_run`), `tools/msckf_sim_eval.cpp`, gate `python/msckf_gates/stage3_sim.py`, tests `test_filter_propagation`, `test_filter_update`.

- **Propagation:** RK4 nominal; Φ in closed form from the preintegrated deltas and their bias Jacobians, with R_k Δv and R_k Δp written as state differences so FEJ evaluates Φ at first estimates only. Q is the preintegration covariance mapped to the global frame plus the bias walk. Φ matches central differences of the RK4 propagation to < 1e-6 over 50 ms. Over 0.5 s the mismatch is 4e-7 to 1.6e-6, falling 4× when the IMU rate doubles (midpoint bias Jacobians vs RK4, second order). Propagation-only NEES (200 runs, 1 s, 15 dof): 14.93 handheld, 14.96 cart, 15.28 fast rotation, band [14.25, 15.77].
- **Cloning:** the time-offset column [ω; v] matches finite differences.
- **Update:** pixel residual through the camera Jacobian, extrinsic blocks; FEJ for clones (rotation and position at cloning). The feature is one triangulated estimate from the current clones, as in MSCKF 2.0 (the yaw and translation nullspace holds for any feature point; an extra FEJ triangulation was tried and made no difference).
- **FEJ vs MSCKF 2.0 (Li & Mourikis, IJRR 2013, `~/work/docs/`):** the paper's attitude error is global (ᴵ_G R ≈ ᴵ_G R̂ (I − ⌊ᴳθ̃×⌋)), so its Φ_θθ = I and it freezes only position and velocity (eqs. 54–56). Ours is body-frame (plan conventions), where the yaw direction R̂ᵀg depends on the rotation estimate, so clone rotations need first estimates too; checked by hand for Φ and H. Using the paper's partial FEJ with our error was tested and is not equivalent (fast-rotation position NEES 10.8). The paper's simulations are a fast car (240 features per image, tracks 4–4.5 frames, 1 px; MSCKF 2.0 NEES 5.89); our slow cart is outside that regime. Its "ideal MSCKF" (Jacobians at the truth) corresponds to `--oracle-jacobians`, which does not fix our cart. The plan's Givens nullspace projection is a Householder QR here (same subspace). Triangulation (`geometry/triangulation`, shared with the init) first used Gauss-Newton on a global xyz point; replaced by anchored inverse depth (decision below). H matches central differences (< 1e-6 relative).
- **Decision: anchored inverse depth for triangulation (user, 2026-10-17).** As in Mourikis & Roumeliotis (ICRA 2007) and the plan. The point is (α, β, ρ) in the first observing camera, p_A = [α, β, 1] / ρ, with the anchor frame turned so its z axis is the anchor's measured bearing: rays past 90° off the optical axis (fisheye) start at α = β = 0, which Mourikis's z = 1 plane cannot represent. Gauss-Newton on the tangent-plane bearing residuals (plan: "on bearings", not normalized-plane pixels as in Mourikis), from the midpoint solution; ρ p_C = R_CA [α, β, 1] + ρ t_CA stays finite as ρ → 0, so far, low-parallax points stay well conditioned. Rejects: rays near parallel (midpoint), point behind the anchor or any camera, ρ ≤ 0. The filter's H_f stays in xyz: H_f^(αβρ) = H_f^(xyz) ∂p/∂(α, β, ρ) with an invertible 3×3 factor, so the left nullspace and the projected update are unchanged. Tests: `test_triangulation` (exact recovery, 120° off axis, 500 m at 0.01° parallax, parallel rays, behind the cameras); a NaN inverse depth is rejected. Benchmark `~/benchmark/2026-10-17_124932` vs `2026-10-12_004139`: handheld bit-identical (same cost, same minimizer); triangulation rejects within 0.01 % (fast rotation +97 of 706k); position NEES shifts within noise (paired per-run differences: fast rotation +0.26 ± 0.16, cart no-FEJ +0.90 ± 0.72, about half the runs worse); Stage 1b still PASS (sim velocity NEES 3.00 → 3.11). Inverse depth does not by itself fix fast rotation or the cart: the 1° parallax gate still drops the far points it handles well. Starting at ρ ≈ 0 when the midpoint lands behind the anchor was considered and skipped: it affects ~100 of 700k features; revisit if the parallax gate is relaxed. Independent review of f94d01c (2026-10-17): no correctness bugs; Jacobian vs central differences 7.8e-10 relative. Open: at low parallax under noise the solver now reaches the true least-squares point, which can be far beyond the true depth (up to 224× in the reviewer's 200 m trials, where the old 5-iteration xyz solve stalled short); the parallax gate looks at bearings, not the estimate, so these points reach the update and the init seeds (`init_refinement` gates at 0.2°). A maximum-depth gate was considered and dropped (user, 2026-10-17): a point at infinity has no depth but still constrains rotation.
- **Decision: features linearized in anchored inverse depth, points at infinity representable (user, 2026-10-17).** `triangulate` returns an `AnchoredPoint` (fixed anchor frame, x = (α, β, ρ)); rays without a midpoint start at infinity instead of being rejected (no baseline leaves ρ at 0); a solution with ρ < 0 (rays meeting behind the cameras) is rejected. A first version clamped ρ to 0 and left such tracks to the χ² gate; the independent review showed the gate cannot see them (the nullspace projection removes exactly the ρ direction: 5.7° of parallax, 23 px raw residual, 0 px projected), so they are rejected in triangulation. `linearizeFeature` works on h = ρ p_C (every camera model is invariant to a positive scale), so at ρ = 0 the clone-position and extrinsic-translation blocks vanish and only rotations are constrained; checked against central differences at finite ρ and at ρ = 0. The xyz Jacobian could not do this: ∂p/∂ρ = −p/ρ diverges at infinity, so the earlier "same nullspace" argument holds for finite points only. The init still needs finite points (ρ > 0).
- **Parallax gate sweep with inverse-depth Jacobians** (`stage3_sim.py --min-parallax`, 50 runs × 60 s, `gate_out/stage3_sim_par1`, `_par05_fix`, `_par0_fix`; ρ < 0 rejected):

  | Gate | fast_rotation final error (FEJ) | fast_rotation position epochs in band, FEJ / no FEJ | cart final error, FEJ / no FEJ | cart orientation epochs in band, no FEJ |
  | --- | --- | --- | --- | --- |
  | 1° (default) | 7.7 cm, 0.57° | 0.6 / 47 % | 17.9 / 0.97 m | 90 % |
  | 0.5° | 6.2 cm, 0.54° | 0.4 / 34 % | 8.0 / 1.8 m | 1 % |
  | 0° | 6.3 cm, 0.53° | 0.4 / 0 % | 22.5 / 2.0 m | 1 % |

  At 1° every number matches the xyz version (finite points: equivalent Jacobians; benchmark `~/benchmark/2026-10-17_163522` identical to `2026-10-17_145125` apart from timing). Below 1°, fast rotation gains accuracy (−19 % position, −7 % rotation error, extrinsics 0.005° / 0.39 mm vs 0.007° / 0.51 mm) but turns overconfident, and the cart's orientation turns inconsistent. The first sweep, with ρ clamped to 0, was worse on the cart with FEJ (170–204 m below 0.5°): the clamped features were the hidden-inconsistency tracks the review found. The simulated scenes have no true points at infinity: low-parallax features are near wall points seen along the motion, and evaluating −ρ J R at a ρ near 0 makes the position update overconfident (the cart failure mode above). **The 1° default stays.** Candidates to use far points safely, untested: a rotation-only update for features whose ρ is consistent with 0; a gate on the triangulated depth uncertainty instead of parallax. Real far points (LaMAria outdoor, KITTI) are where the benefit should show.
- **Independent review of 0ad58af (2026-10-17):** the ρ < 0 rejection is complete. Found: a track with no baseline and inconsistent bearings gets ρ = 0, so H_f's ρ column is zero and the unpivoted Householder QR discarded a left-nullspace row that held the inconsistency (40 px became 0.05 px). Fixed: `projectOutFeature` uses column-pivoted QR and keeps 2M − rank(H_f) rows, with one χ² degree of freedom each. Open (also in the xyz version): Gauss-Newton's 5 iterations have no convergence check, so an unconverged solve can be accepted; seen only at outlier-level noise (0.02 rad), where the χ² gate should catch it.
- **Window:** a track is used when it ends or when the oldest of `max_clones` (11) clones goes; a landmark seen after its track was used starts a new track (marginalized independently, no double counting).
- **Gate `stage3_sim` (50 runs × 60 s, 1 px; gated: handheld, fast_rotation; cart reported): FAIL.** Handheld consistent with and without FEJ (orientation / position NEES 3.11 / 3.12, band [2.36, 3.72]), 4.9 cm final error RMSE, 3.3 ms per image; extrinsic rotation 0.025°, t_d 0.03 ms pass; extrinsic translation 3.1 mm fails (target 1 mm). Fast rotation: all calibration passes (0.5 mm, 0.007°, 0.014 ms) but position NEES 6.14 with FEJ (3.71 without); cart position NEES 139 with FEJ, 7.8 without (final run on the staged code, `gate_out/stage3_sim/report.md`). Under the gate's per-epoch criterion (average NEES inside the band in ≥ 90 % of epochs, adopted in review): handheld orientation / position 97 / 100 % with FEJ (99 / 100 % without) pass; fast rotation 85 / 1 % with FEJ and 84 / 53 % without, so fast rotation is inconsistent in position with or without FEJ, worse with it. **Open: FEJ raises position NEES where no-FEJ does not** (theory says the opposite); next is the plan's observability metric (numerical rank and ‖O·N‖ with FEJ on and off) to find the cause.
- **Consistency criterion changed (code review, 2026-10-12):** the run-long mean NEES was checked against the band for one epoch's 50-run average, far too wide for a mean over ~1200 epochs. The gate now requires the per-epoch average inside that band in ≥ 90 % of epochs (95 % expected for independent epochs; slack for correlation); the run-long mean is reported. Re-scored on the run above: handheld 97 / 100 % (orientation / position) with FEJ, 99 / 100 % without, pass; fast rotation 85 / 0.6 % with FEJ (53 % position without), fail; cart 34 / 0.9 %. The gate also refuses missing runs or a run > 1 % shorter than the others instead of truncating all runs to it.
- **Simulation, 20 runs × 30 s, 1 px (first look):** handheld consistent with and without FEJ (orientation / position NEES 2.94 / 3.39, band [2.02, 4.17]); 3.6 cm final error; 2.9 ms per image median. Extrinsic rotation and t_d converge (0.04°, 0.04 ms); extrinsic translation 3.7 mm after 30 s (target 1 mm).
- **Cart (simulation) is overconfident:** position NEES 20.5 with FEJ, 6.7 without, ~1 m drift in 30 s, along the horizontal velocity (scale); biases and orientation stay consistent; the propagation alone is consistent. Ruled out with `msckf_sim_eval` diagnostics: calibration (same with extrinsic and t_d fixed), triangulation convergence (30 iterations: identical), FEJ point mismatch, bookkeeping (30 clones fine on handheld), per-feature NIS (0.97 per row). Found: features accepted by the 1° parallax gate have a depth bias (log ratio −0.06 at the start, before drift; handheld 0.000), from selecting on noisy parallax; selecting on the true parallax removes the bias but not the overconfidence. The cart's features are low-parallax (depth error std 20 % vs 6 % handheld); without the gate it diverges, with a stricter one too few survive. Inflating the filter's pixel noise 3× makes it nearly consistent (pose NEES 3.6–8.5). Working explanation: second-order error of poorly triangulated points, coherent across a batch under forward motion. Candidate fixes: clones at keyframes (parallax-spaced) instead of every image, an unbiased feature-quality test, a ZUPT for the stops. Deprioritized (user, 2026-10-11).

### Equation audit vs the plan (2026-10-17)

Every equation in the plan artifact (Conventions, Double Sphere, Stage 1a, Stage 3) was checked against the code at 8debcea. Audit only; no code changed.

**Matches:** Hamilton R_GI with right perturbation for the IMU, the extrinsic and the clones; additive v, p errors in G; g = [0, 0, −9.81]; aₘ = R_GIᵀ(a − g) + b_a in the simulator and RK4; t_imu = t_cam + t_d. The closed-form Φ agrees with the continuous error dynamics above (Φ_θθ = R_{k+1}ᵀR_k, Φ_θ,bg = ∂ΔR/∂b_g, Φ_vθ = −⌊R_kΔv×⌋R_k, Φ_v,ba = R_k ∂Δv/∂b_a, Φ_pv = T·I). Clone Jacobian: identity on δθ, δp and the [ω; v] time-offset column at t_cam + t_d. H clone and feature blocks as in step 2 above; extrinsic blocks ⌊ᶜp×⌋ and −R_ICᵀ. χ² gate at 95 % with 2M − 3 dof, QR compression, Joseph form, injection R ← R̂ Exp(δθ). Double Sphere projection. Stage 1a gravity, biases and yaw gauge.

**Deviations:**

| # | Plan | Code | Where | Status |
| --- | --- | --- | --- | --- |
| 1 | Error-state order [δθ, δb_g, δv, δb_a, δp] | [δθ, δp, δv, δb_g, δb_a] | `include/msckf/init/imu_init_state.h` (`imu_err`) | Open |
| 2 | Solà's reset Jacobian G = I − ½⌊δθ×⌋ applied to P after injection | No reset | `FilterState::inject`, `featureUpdate` | Open (second order; untested) |
| 3 | MSCKF 2.0 closed-form Φ and continuous-time Q | Φ from the preintegrated deltas and bias Jacobians; Q = preintegration covariance mapped to G plus bias walk, without bias cross-coupling | `src/filter/propagator.cpp` | Equivalent to first order; Φ matches finite differences (see Progress) |
| 4 | Every Φ and H at first estimates | Clone poses and IMU p, v, R frozen. Not frozen: H's extrinsic blocks (current R_IC, p_IC), Φ's bias Jacobians (current biases). The IMU linearization point is the last propagated estimate (standard FEJ) | `linearizeFeature`, `imuTransition` | Open: relevant to the FEJ NEES question; the extrinsic does not enter the yaw / position nullspace |
| 5 | Anchored inverse depth triangulation | Was Gauss-Newton on a global xyz point. Cost is bearing residuals (plan), not Mourikis's normalized-plane pixels | `src/geometry/triangulation.cpp` | Fixed 2026-10-17: anchored inverse depth (Progress, decision) |
| 6 | Givens nullspace projection | Householder QR | `featureUpdate` | Justified: same subspace |
| 7 | Use every track touching the oldest clone before marginalizing | Tracks with < 3 observations (`min_observations`) drop their observation in that clone unused | `Msckf::addImage` | Open (minor) |
| 8 | Zero-velocity update | Not implemented | — | Open (Handover, item 3) |
| 9 | FEJ and right-invariant SE₂(3) behind one interface | FEJ multiplicative error only | — | Open (Handover, item 3) |

The Hamilton-reference and error-state-order paragraphs under Conventions were added to the artifact after this file was last synced, so items 1 and 2 were not visible here when the filter was written.

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
| KITTI raw (unsynced extract) | Pinhole + radtan (k3) | Car, 10 Hz camera, 100 Hz OXTS | OXTS GPS/INS (fused velocity) | Car-like init, **diagnosis only**: no camera–IMU sync, filtered OXTS "IMU", timestamp jumps and gaps; the init's sim car is consistent where KITTI is not |
| Zurich Urban MAV (AGZ) | Pinhole + radtan (k1 k2 p1 p2 k3, `calibration_data.npz`) | Tethered MAV flying 5–15 m over Zurich streets, ~2 km, ~45 min | Photogrammetric camera poses (Street View aligned, UTM 32N) for every 30th image only | Aerial urban front-end data; weak for VIO (below) |
| Exodigo cart (own) | ZED-X left (pinhole), one Insta360 lens (DS) | Slow, planar, frequent stops | RTK GPS | The target system |

**Zurich Urban MAV (AGZ), as found on disk (2026-10-11; not integrated, user paused it for the handover):**
- **Layout:** `/mnt/c/Users/user/data/zurich_mav/AGZ/` holds 81 169 JPEGs (`MAV Images/00001.jpg`…, 1920×1080 colour GoPro, ~30 Hz).
- **Calibration:** `calibration_data.npz`: fx 893.4, fy 898.3, cx 951.1, cy 555.1; distortion (−0.2805, 0.1158, −0.00098, 0.00016, −0.0270); from 30 images in `MAV Images Calib/`.
- **Logs (`Log Files/*.csv`, timestamps in µs):**
  - `RawGyro.csv` and `RawAccel.csv` run at only **10 Hz**.
  - `OnboardPose.csv` (Pixhawk) runs at 50 Hz: rates, accelerations, its velocity, accelerometer bias and attitude, plus tether data.
  - `OnboardGPS.csv` has one row per image (imgid and timestamp: the image clock).
  - `GroundTruthAGL.csv` has 2 708 poses, every 30th image (~1 Hz): UTM position and photogrammetric angles omega/phi/kappa in degrees, plus the raw GPS.
- **Also present:** `AGZ.bag` (212 MB) and `air-ground-data/` (the IROS13 air-ground matching subset), plus the papers under `zurich_mav/papers/`.
- **Assessment:**
  - Good for the front end: urban aerial imagery at 30 Hz.
  - Poor for VIO and init: the IMU is 10 Hz raw or 50 Hz filtered, there is **no camera–IMU extrinsic or time-offset calibration**, the GoPro is rolling-shutter, and ground truth is 1 Hz.
  - Before use: a converter to the EuRoC layout (like `kitti_raw_to_euroc.py`), probably downscaled to 960×540, and an extrinsic estimated or taken from the IJRR17 paper. Then front-end metrics (track length, spread) against TartanAir and EuRoC. The repo needs a committed fetch script (`scripts/data/`), per the no-throwaway-scripts rule.

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
- **GTSAM:** preintegration math and Lie-group utilities, as a cross-check for the initialization factors. Built once from the local checkout by `scripts/deps/build_gtsam.sh` (system Eigen, no -march=native, no Boost/TBB) and found with `-DGTSAM_DIR`; only `test_preintegration_gtsam` links it until Stage 4.
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
