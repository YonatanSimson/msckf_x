#pragma once

#include "msckf/init/featureless_init.h"
#include "msckf/init/init_refinement.h"
#include "msckf/init/static_init.h"

#include <optional>
#include <string>
#include <vector>

namespace msckf {

// Stage 1b pipeline (plan, "Dynamic initialization", steps 1-3): from a start
// time, take the window of frames, reduce it to keyframes (selectKeyframes),
// solve the feature-less linear system and refine it. If any step refuses
// (excitation, conditioning, too few pairs or features, no convergence), the
// window grows by `growth` and the attempt is repeated, up to `max_window`
// (step 3: "retry with a longer window"). Mono scale is observable only
// through acceleration, so a cart or car cruising at constant speed needs the
// window to reach a speed change or a turn.
// Scale gate: a solution is accepted only if its hand-off velocity sigma
// (largest eigenvalue) is below max(max_relative_velocity_sigma * speed,
// min_velocity_sigma). Mono scale collapses in weakly accelerated windows (the
// scene and trajectory shrink together and still fit), and the absolute
// covariance at a collapsed scale looks confident; the relative one does not.
// Threshold 30 % (2026-10-11): on simulation (handheld, car, cart, fast
// rotation; 0.5-1 s windows) solutions below 30 % were consistent (velocity
// NEES 2.0-2.9) and those above mostly had collapsed scale (NEES 6-20, scale
// off by > 20 % in 52 of 73); on EuRoC and TUM-VI solutions above 30 % were
// still accurate (velocity error 0.13 m/s, NEES 2.5), so 10 % refused good
// real-data solutions. This replaces
// the linear step's accelerometer-excitation threshold, which the paper does
// not have and which refused most cart and car windows outright.
// Static first: each attempt first checks whether the device is at rest over
// the attempt's window (its last min(window, static_init.window) seconds):
// accelerometer spread and the feature disparity (still_disparity_quantile),
// from the bearings
// with pixels_per_radian. If it is, static initialization (Stage 1a: gravity
// from the bias-corrected mean specific force, v = 0) is the hand-off. A
// still device has no parallax, so the dynamic steps cannot succeed there
// (they refuse with "features"), and need not: at rest nothing is unknown
// but the tilt.
// In place: a handheld device held "still" still rotates (TUM-VI room1
// starts with 1-6 deg/s of hand wobble, 0.1-0.4 m/s^2 of tremor and 2-8 px
// of raw disparity per 0.5 s, but 1-3 cm/s of translation), which fails both
// the static tests and the dynamic steps (no parallax). Without translation
// gravity is still the mean specific force, once each sample is rotated into
// one frame by the gyro, and v = 0. Accepted when the rotation-compensated
// disparity (still_disparity_quantile) is below in_place_max_disparity_px (no translation
// relative to the scene) and the mean specific force has gravity's norm.
// Refinement policy. The paper: refinement is the primary path, and "it is
// possible to skip iterative refinement and use a conservative initial
// covariance", initializing in 0.6 ms (kNever, its "Ours (NR)"). kFallback
// (ours): refine, and when the refinement refuses, hand off the linear
// solution instead (linearHandoff). Either way the scale gate applies.
enum class InitRefinementPolicy { kAlways, kFallback, kNever };

struct DynamicInitConfig {
  InitRefinementPolicy refine = InitRefinementPolicy::kAlways;
  // Linear hand-off covariance = linear-solve covariance x this (variance
  // factor; the paper's "conservative").
  double linear_covariance_scale = 1.0;
  bool try_static = true;
  StaticInitConfig static_init;
  // Bearing angle to pixels for the disparity test (the focal length). 0: no
  // image test, the IMU test only (passes a cart cruising at constant velocity).
  double pixels_per_radian = 0.0;
  // Quantile of the feature disparities tested for stillness (static and in
  // place). Not the median: outdoors most features are far away and barely
  // move while the device translates; any near feature that moves proves
  // translation (sim cart, slow U-turns at 0.2-0.5 m/s: the median passed,
  // the 90th percentile does not).
  double still_disparity_quantile = 0.9;
  bool try_in_place = true;  // needs pixels_per_radian > 0
  double in_place_max_disparity_px = 1.0;
  // Velocity sigma of the in-place hand-off (v = 0): the residual
  // translation of a hand held in place (TUM-VI room1: 1-3 cm/s).
  double in_place_velocity_sigma = 0.05;  // m/s
  double min_window = 0.5;  // s; 0.3 s monocular windows are unreliable (plan)
  double growth = 0.25;     // s added per retry
  double max_window = 5.0;  // s
  double max_relative_velocity_sigma = 0.30;
  // Absolute floor (m/s). 0: none. A collapsed scale looks like standing
  // still (cart turning at 0.5 m/s estimated at 0.02 m/s with sigma < 0.05),
  // so a floor admits it; stationary starts belong to static init (Stage 1a,
  // which also checks the image is still).
  double min_velocity_sigma = 0.0;
  // Windows whose IMU stream has a gap above this are refused ("imu_gap"):
  // preintegration would interpolate across missing data (KITTI's OXTS drops
  // up to 0.55 s).
  double max_imu_gap = 0.05;  // s
  FeaturelessInitConfig linear = [] {
    FeaturelessInitConfig c;
    c.min_acc_excitation = 0.0;  // replaced by the scale gate
    return c;
  }();
  InitRefinementConfig refinement;
};

struct DynamicInitResult {
  bool ok = false;
  bool is_static = false;  // the hand-off is static initialization's (refinement.state; no clones)
  bool in_place = false;   // ... or in-place (rotating, not translating; also is_static)
  bool linear_only = false;  // the hand-off is the linear solution (linearHandoff; no clones)
  std::string reason;   // of the last attempt, when not ok ("scale": failed the scale gate)
  double velocity_sigma = 0.0;  // m/s, hand-off, of the successful (or last refined) attempt
  double window = 0.0;  // s, of the successful (or last) attempt
  int attempts = 0;
  std::vector<double> keyframe_t;  // times of the keyframes of the successful (or last) attempt
  FeaturelessInitResult linear;      // of the successful (or last) attempt
  InitRefinementResult refinement;   // idem; refinement.state is the hand-off
};

// The scale gate on a refined hand-off: its velocity sigma (largest
// eigenvalue, also returned) against max(relative * speed, floor).
bool passesScaleGate(const ImuInitState& state, const DynamicInitConfig& config, double* velocity_sigma = nullptr);

// In-place initialization over frames (first to last): the gyro-compensated
// mean specific force gives the tilt, v = 0. nullopt unless the
// rotation-compensated disparity quantile (px, returned) and the specific
// force norm pass.
std::optional<ImuInitState> inPlaceInitialize(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                              const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                              const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                              const DynamicInitConfig& config, double* disparity_px = nullptr);

// Hand-off from the linear solution alone (the paper's NR): the state at the
// last keyframe, propagated from (v_I0, g_I0) by IMU preintegration, with the
// first-order covariance of the linear solve (FeaturelessInitResult::covariance,
// times linear_covariance_scale) mapped to tilt, position and velocity, plus
// the preintegration noise and the bias priors. The global frame's shift with
// the gravity error is ignored (first order in the tilt, covered by the scale).
// nullopt if the linear result has no covariance.
std::optional<ImuInitState> linearHandoff(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& keyframes,
                                          const FeaturelessInitResult& linear, const Eigen::Vector3d& gyro_bias,
                                          const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                          const DynamicInitConfig& config);

// frames: all frames available from t_start on (more than max_window is
// fine); imu must cover them.
DynamicInitResult initializeDynamic(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                    double t_start, const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                    const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                    const DynamicInitConfig& config = {});

}  // namespace msckf
