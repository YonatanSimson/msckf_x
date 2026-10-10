#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <vector>

namespace msckf {

// Pose and derivatives of the IMU at one time, all exact.
struct SplineState {
  Sophus::SO3d R_GI;          // IMU to global
  Eigen::Vector3d p_GI;       // IMU position in G
  Eigen::Vector3d v_GI;       // velocity in G
  Eigen::Vector3d a_GI;       // acceleration in G
  Eigen::Vector3d omega_I;    // angular velocity in the IMU frame
  Eigen::Vector3d alpha_I;    // angular acceleration in the IMU frame
};

// C2 uniform cubic B-spline over the split pose space SO(3) x R^3, both parts in
// cumulative form. Rotation: R(u) = R_i prod_j Exp(B~_j(u) d_j) with
// d_j = Log(R_{i+j-1}^-1 R_{i+j}); translation the same with vector differences.
// Body angular velocity and acceleration follow from the product's recursion, so
// the IMU signal is analytic. The curve approximates its control poses; it does
// not pass through them.
class SplineTrajectory {
 public:
  SplineTrajectory(double t0, double knot_dt) : t0_(t0), dt_(knot_dt) {}

  void addControlPose(const Sophus::SO3d& R_GI, const Eigen::Vector3d& p_GI);

  int numControlPoses() const { return static_cast<int>(rotations_.size()); }
  double knotInterval() const { return dt_; }
  double minTime() const { return t0_; }
  double maxTime() const { return t0_ + (numControlPoses() - 3) * dt_; }

  // Requires minTime() <= t <= maxTime() and at least 4 control poses.
  SplineState evaluate(double t) const;

 private:
  double t0_;
  double dt_;
  std::vector<Sophus::SO3d> rotations_;
  std::vector<Eigen::Vector3d> positions_;
};

}  // namespace msckf
