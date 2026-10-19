#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <memory>

namespace msckf {

/// Pose and derivatives of the IMU at one time, all exact.
struct TrajectoryState {
  Sophus::SO3d R_GI;          ///< IMU to global
  Eigen::Vector3d p_GI;       ///< IMU position in G
  Eigen::Vector3d v_GI;       ///< velocity in G
  Eigen::Vector3d a_GI;       ///< acceleration in G
  Eigen::Vector3d omega_I;    ///< angular velocity in the IMU frame
  Eigen::Vector3d alpha_I;    ///< angular acceleration in the IMU frame
};

/// A simulated IMU motion with analytic derivatives, so the IMU signal is exact.
class Trajectory {
 public:
  virtual ~Trajectory() = default;

  /// Exact state at time t, for minTime() <= t <= maxTime().
  virtual TrajectoryState evaluate(double t) const = 0;
  /// First time the trajectory is defined at (s).
  virtual double minTime() const = 0;
  /// Last time the trajectory is defined at (s).
  virtual double maxTime() const = 0;
};

using TrajectoryPtr = std::shared_ptr<const Trajectory>;

}  // namespace msckf
