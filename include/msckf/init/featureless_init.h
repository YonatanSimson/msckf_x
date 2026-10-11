#pragma once

#include "msckf/imu/imu_types.h"
#include "msckf/init/imu_init_state.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace msckf {

// Stage 1b, step 1: feature-less dynamic initialization (plan, "Dynamic
// initialization"), from the sqrtVINS paper (Peng, Chen, Wu, Huang, T-RO 2025,
// Sec. V-A, Propositions 1-2; paper only, never its code).
//
// Unknowns: the velocity v and gravity g of the first frame's IMU, both in
// that IMU frame {I0}; no 3D features. With IMU preintegration from t_0 to
// each frame k (dR_k, alpha_k = dp_k, dT_k; biases taken as known, as in the
// paper), and the camera pose p_IC, R_IC in the IMU frame:
//
//  - Camera-induced direction (Prop. 1): every feature seen in frames j and k
//    gives the normal n = [b_j]x b_k of its epipolar plane, with bearings
//    rotated into {I0} (b = dR_k R_IC b_cam). The direction t of the camera
//    translation from j to k is the eigenvector of M = sum n n^T with the
//    smallest eigenvalue; e = [e1 e2] spans the plane orthogonal to t.
//  - IMU-induced displacement (Prop. 2, eq. 60): the same camera translation
//    is s t = (dT_k - dT_j) v + 1/2 (dT_k^2 - dT_j^2) g + (alpha_k - alpha_j)
//    + (dR_k - dR_j) p_IC, with unknown scale s (g is the gravity vector, as in
//    preintegration.h; the paper's eq. 60 writes the (v, g) block as -A).
//    Projecting onto e removes s: two linear equations in x = (v, g) per
//    frame pair.
//
// Decisions the paper leaves open (recorded in the plan):
//  - All pairs of the given keyframes are used (the paper states O(k^2) in the
//    number of keyframes); pairs with fewer than min_common_features common
//    features are skipped. Normals are not normalized, so near-parallel
//    bearings (little parallax) weigh little.
//  - The 6-unknown least-squares solution is followed by Gauss-Newton on the
//    same residuals with |g| fixed (g on its 2-sphere, as the paper's
//    (alpha, beta) parameterization), so the result has exact gravity norm.
//  - Excitation gate (plan, step 3): the smallest singular value of the
//    stacked system with unit-norm columns, and the accelerometer excitation,
//    each above a threshold. Excitation is the spread of the specific force
//    rotated into {I0} (sqrt of the trace of its sample covariance): gravity
//    is constant there, so only real accelerations raise it. The caller
//    retries with a longer window on rejection.

// One frame's tracked features: id and unit bearing in the camera frame.
struct InitFrame {
  double t = 0.0;  // IMU time of the image (camera time + time offset)
  std::vector<std::pair<int64_t, Eigen::Vector3d>> bearings;
};

struct FeaturelessInitConfig {
  int min_common_features = 10;      // per keyframe pair
  int min_pairs = 2;                 // usable keyframe pairs
  double min_singular_value = 1e-3;  // of the system with unit-norm columns
  double min_acc_excitation = 0.2;   // m/s^2, specific force spread in {I0}
  int gravity_iterations = 5;        // Gauss-Newton with |g| fixed
  double gravity_norm = 9.81;        // m/s^2 (constants.h)
  // Relative-scale equations from features seen in three keyframes (ours, not
  // in the paper). Prop. 2 removes each pair's scale s_ij, so on a straight
  // path (all translations parallel) the speed along it drops out and the
  // solve returns ~0 for it. A feature seen from keyframes i, j, k has one
  // depth at i: triangulated with unit baselines along t_ij and t_ik it gives
  // mu_ij, mu_ik with s_ij mu_ij = s_ik mu_ik, so with rho = median over the
  // features of mu_ik / mu_ij, one more equation per (i, j, k):
  //   (t_ij^T (A_ij x + rb_ij) - rho t_ik^T (A_ik x + rb_ik)) / sqrt(1 + rho^2) = 0,
  // linear in x. It restores the along-path speed whenever the speed changes
  // over the window (constant velocity stays unobservable). Translation
  // directions get their sign from positive depths.
  // Off (2026-10-11): on the linear solve alone they help (sim cart, 1 s:
  // gravity 2.35 -> 1.08 deg at 1 px noise; at 0.3 px speed error 53 -> 18 %,
  // noise-free 50 -> 7 %), but through the full retry pipeline the
  // refinement already recovers the speed from its landmarks, and the extra
  // equations sometimes hand it a worse start it then accepts confidently
  // (sim cart scale RMSE 28 -> 119 %, velocity NEES 5.2 -> 93; car 24 -> 43 %;
  // handheld and fast rotation unchanged). Re-test on real cart data
  // (dynamic_init_eval --triplets).
  bool triplet_scale = false;
  // Per feature and pair, rotation-compensated. 2 deg (sim, 1 px noise): 0.5
  // deg let noisy depth ratios through (handheld 0.5 s gravity 2.2 deg vs 1.5).
  double triplet_min_parallax_deg = 2.0;
  int triplet_min_features = 5;           // per triplet
  double triplet_weight = 1.0;            // relative to the epipolar rows
};

struct FeaturelessInitResult {
  bool ok = false;
  std::string reason;  // why not ok: "pairs", "excitation", "conditioning", "imu"
  Eigen::Vector3d v_I0 = Eigen::Vector3d::Zero();  // velocity of the first IMU frame, in it
  Eigen::Vector3d g_I0 = Eigen::Vector3d::Zero();  // gravity in the first IMU frame
  double singular_value = 0.0;  // smallest, of the system with unit-norm columns
  double acc_excitation = 0.0;  // m/s^2
  int pairs = 0;                // keyframe pairs used
  int triplets = 0;             // relative-scale equations used (triplet_scale)
  // Per used pair (j, k): the translation direction's eigenvalue ratio
  // lambda_t / lambda_1 (small = well-defined direction) and the scale s
  // implied by the solution (camera translation length, m).
  std::vector<std::pair<int, int>> pair_frames;
  std::vector<double> pair_eigen_ratio;
  std::vector<double> pair_scale;
  // Covariance of the solution, x = (v_I0, gravity direction on its tangent
  // plane: g_I0 = |g| normalize(u + B dx)), from the Gauss-Newton normal
  // equations with the residual variance estimated from the fit (floored by
  // the IMU's preintegrated position noise). The paper's "conservative
  // initial covariance" for skipping refinement gives no values; this makes
  // the degenerate directions visible: on a straight path every pair's
  // translation is parallel, the projection removes the speed along it, and
  // its variance is huge instead of the estimate silently being ~0.
  bool has_covariance = false;
  Eigen::Matrix<double, 5, 5> covariance = Eigen::Matrix<double, 5, 5>::Zero();
  Eigen::Matrix<double, 3, 2> gravity_basis = Eigen::Matrix<double, 3, 2>::Zero();  // B
};

// Keyframes of a window, as in the paper ("selected based on the average
// parallax"; 3 for a 0.1 s window, 5 for 0.5 s): indices into frames, the
// first and last always included, at most max_keyframes, spaced evenly in the
// cumulative mean rotation-compensated parallax between consecutive frames
// (bearings rotated by the gyro into a common frame, so rotation alone adds
// none). With max_keyframes or fewer frames, all are used.
std::vector<int> selectKeyframes(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                 const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias, int max_keyframes = 5);

// Solves for (v, g) from the frames (first frame = {I0}; at least two) and the
// IMU samples covering [frames.front().t, frames.back().t].
FeaturelessInitResult featurelessInit(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                      const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                      const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                      const FeaturelessInitConfig& config = {});

}  // namespace msckf
