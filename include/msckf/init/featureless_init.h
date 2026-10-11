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
};

struct FeaturelessInitResult {
  bool ok = false;
  std::string reason;  // why not ok: "pairs", "excitation", "conditioning", "imu"
  Eigen::Vector3d v_I0 = Eigen::Vector3d::Zero();  // velocity of the first IMU frame, in it
  Eigen::Vector3d g_I0 = Eigen::Vector3d::Zero();  // gravity in the first IMU frame
  double singular_value = 0.0;  // smallest, of the system with unit-norm columns
  double acc_excitation = 0.0;  // m/s^2
  int pairs = 0;                // keyframe pairs used
  // Per used pair (j, k): the translation direction's eigenvalue ratio
  // lambda_t / lambda_1 (small = well-defined direction) and the scale s
  // implied by the solution (camera translation length, m).
  std::vector<std::pair<int, int>> pair_frames;
  std::vector<double> pair_eigen_ratio;
  std::vector<double> pair_scale;
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
