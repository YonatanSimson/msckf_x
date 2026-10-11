#pragma once

#include "msckf/imu/imu_types.h"
#include "msckf/init/featureless_init.h"
#include "msckf/init/imu_init_state.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <string>
#include <vector>

namespace msckf {

// Stage 1b, step 2: refinement of the feature-less solution (plan, "Dynamic
// initialization"; sqrtVINS Sec. V-B, paper only), as an iterated EKF update
// in information form rather than the paper's square-root filter.
//
// State: one clone (R_GI, p, v) per frame of the window, plus gyro and
// accelerometer biases shared by all (constant over a window of <= 1 s),
// errors in the filter's convention (R = R_hat Exp(dtheta) with dtheta in the
// IMU frame; true = est + err otherwise). The global frame is the first
// frame's IMU, gravity aligned, yaw 0 (attitudeFromUp), origin at its
// position. Start: the first clone from the linear solve, the others
// propagated from it.
//
// Landmarks are in the state too, as the paper's key features: every feature
// seen in >= 2 frames with enough parallax, triangulated once from the
// initial clones (global position), then estimated jointly. Every iteration
// relinearizes all terms and takes a Gauss-Newton step, the landmarks
// eliminated by the Schur complement on their 3x3 blocks and updated by
// back-substitution:
//  - prior on the first clone (loose tilt and velocity, tight yaw and
//    position: the gauge) and on the biases (their prior calibration);
//  - IMU: the preintegration between consecutive clones as a 9-dim residual
//    (Forster et al.) with its covariance, bias-corrected to first order;
//  - vision: each landmark's tangent-plane bearing residuals in every frame
//    that sees it. Chi-square gate in two phases: iterate with all landmarks
//    until the step is below gate_convergence (1e-3, close enough to tell
//    inliers from outliers), decide once there, then converge with the
//    inliers.
// Convergence: largest step below 1e-5 (rad, m, m/s; landmarks relative to
// their distance). The hand-off is the IMU state at the last frame with the
// landmarks and their joint covariance (from the Schur complement); the
// filter uses these landmarks only until they are lost (user, 2026-10-11:
// features in the state like sqrtVINS, only for initialization).
//
// Decided in implementation (recorded in the plan):
//  - The IMU enters as a relinearized measurement, not as a prior propagated
//    once: with a once-propagated prior a 5 deg start stalled at 0.17 deg
//    without noise; now it converges to 1e-4 deg, landmarks to 2e-5 m.
//  - Keyframes (selectKeyframes, at most 5), as in the paper.
//  - Monocular windows of 0.3 s are not reliable on simulation; with the
//    landmarks in the state, most of the wrong solutions no longer converge
//    and are refused. 0.5 s and 1 s windows are accurate and consistent (IMU
//    and landmark NEES in their 95% bands).
struct InitRefinementConfig {
  // Priors on the first clone's roll, pitch and velocity are centred on the
  // linear solution, which comes from the same measurements: they must be
  // uninformative, or the refinement double-counts them (with 1 m/s, a car at
  // 8 m/s settled near the linear velocity and reported 3-5x too small a
  // velocity sigma).
  double tilt_sigma = 0.5;          // rad
  double velocity_sigma = 10.0;     // m/s
  double gauge_sigma = 1e-4;        // rad / m: yaw and position of the first clone
  double gyro_bias_sigma = 0.005;   // rad/s, prior calibration
  double acc_bias_sigma = 0.1;      // m/s^2, prior calibration
  double bearing_sigma = 1.0 / 400.0;  // rad, about 1 px at f = 400 px
  double min_parallax_deg = 0.2;    // features below are not used
  double chi2_quantile = 0.95;      // per-feature gate
  int max_iterations = 30;          // both phases (before and after the gate)
  double convergence = 1e-5;        // rad, m, m/s: max |Gauss-Newton step|
  double gate_convergence = 1e-3;   // phase 1: close enough to decide the gate
};

struct InitRefinementResult {
  bool ok = false;     // converged with a positive-definite covariance
  std::string reason;  // "features", "diverged", "covariance"
  int iterations = 0;
  std::vector<double> steps;  // max |correction| of each iteration (diagnostics)
  // Whitened squared residuals (prior, IMU, vision) at the last iterate, and
  // the number of residual rows minus states: cost / dof ~ 1 for a consistent
  // fit.
  double cost = 0.0;
  int dof = 0;
  int features = 0;    // used in the last iterate
  int rejected = 0;    // by the chi-square gate in the last iterate
  ImuInitState state;  // at the last frame
  // Refined clones, one per frame (global frame above), and the biases.
  std::vector<NavState> clones;
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
  // Landmarks handed to the filter with the IMU state (the paper's key
  // features), and the joint covariance of [the IMU error of `state` (15,
  // imu_err order), each landmark's position (3, global frame, in this
  // order)].
  struct Landmark {
    int64_t id = 0;
    Eigen::Vector3d p_G = Eigen::Vector3d::Zero();
  };
  std::vector<Landmark> landmarks;
  Eigen::MatrixXd P_joint;
};

InitRefinementResult refineInit(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                const Sophus::SE3d& T_IC, const FeaturelessInitResult& linear,
                                const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias,
                                const ImuNoise& noise, const InitRefinementConfig& config = {});

}  // namespace msckf
