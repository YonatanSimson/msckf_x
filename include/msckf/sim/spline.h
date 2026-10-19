#pragma once

#include "msckf/sim/trajectory.h"

#include <vector>

namespace msckf {

// C2 uniform cubic B-spline over the split pose space SO(3) x R^3, both parts in
// cumulative form. Rotation: R(u) = R_i prod_j Exp(B~_j(u) d_j) with
// d_j = Log(R_{i+j-1}^-1 R_{i+j}); translation the same with vector differences.
// Body angular velocity and acceleration follow from the product's recursion, so
// the IMU signal is analytic. The curve approximates its control poses; it does
// not pass through them.
class SplineTrajectory : public Trajectory {
 public:
  SplineTrajectory(double t0, double knot_dt) : t0_(t0), dt_(knot_dt) {}

  void addControlPose(const Sophus::SO3d& R_GI, const Eigen::Vector3d& p_GI);

  int numControlPoses() const { return static_cast<int>(rotations_.size()); }
  double knotInterval() const { return dt_; }
  double minTime() const override { return t0_; }
  double maxTime() const override { return t0_ + (numControlPoses() - 3) * dt_; }

  // Requires minTime() <= t <= maxTime() and at least 4 control poses.
  TrajectoryState evaluate(double t) const override;

 private:
  double t0_;
  double dt_;
  std::vector<Sophus::SO3d> rotations_;
  std::vector<Eigen::Vector3d> positions_;
};

}  // namespace msckf
