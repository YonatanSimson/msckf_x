#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace msckf::vision {

// Two-view outlier rejection on unit bearings, valid for any field of view
// (bearings past 90 degrees included), with the linear 8-point essential matrix
// f_b^T E f_a = 0 as the minimal model and the angular residual of each bearing
// to its epipolar plane (the larger of the two directions).
//  - kLoMsac (default): MSAC scoring (truncated quadratic, Torr & Zisserman
//    2000) and local optimization of every new best model by iterated,
//    reweighted least squares on the inliers of a shrinking threshold
//    (LO-RANSAC, Lebeda, Matas & Chum 2012).
//  - kRansac: inlier counting and one final refit, for comparison.
// With pure rotation every E = [t]x R fits the inliers, so they are still all
// accepted.
enum class RobustMethod { kRansac, kLoMsac };

struct BearingRansacConfig {
  RobustMethod method = RobustMethod::kLoMsac;
  double threshold_rad = 1e-3;  // max angle to the epipolar plane
  int max_iterations = 500;
  double confidence = 0.999;
  uint64_t seed = 1;
};

struct BearingRansacResult {
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  std::vector<bool> inlier;
  int num_inliers = 0;
  int iterations = 0;
  bool ok = false;  // false if fewer than 8 correspondences
};

BearingRansacResult bearingRansac(const std::vector<Eigen::Vector3d>& f_a,
                                  const std::vector<Eigen::Vector3d>& f_b,
                                  const BearingRansacConfig& config = {});

// Angle between bearing f_b and the epipolar plane of f_a under E, and vice
// versa; returns the larger (radians).
double epipolarAngle(const Eigen::Matrix3d& E, const Eigen::Vector3d& f_a, const Eigen::Vector3d& f_b);

}  // namespace msckf::vision
