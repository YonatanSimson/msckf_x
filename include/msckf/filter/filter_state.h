#pragma once

#include "msckf/imu/imu_types.h"
#include "msckf/init/imu_init_state.h"

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <deque>

namespace msckf::filter {

// What the filter estimates besides the IMU state and the clones.
struct StateOptions {
  bool fej = true;                  // first-estimates Jacobians (plan, Stage 3 "Propagation")
  bool estimate_extrinsic = true;   // R_IC, p_IC
  bool estimate_time_offset = true;  // t_d, with t_imu = t_cam + t_d
};

// An IMU pose cloned at an image time. The FEJ copy is the estimate at the
// moment of cloning, before any update touched it. Rotation needs a first
// estimate too: our attitude error is in the body frame, where the yaw
// direction R^T g depends on the rotation estimate (MSCKF 2.0 uses a global
// attitude error and freezes only position and velocity).
struct Clone {
  double t_cam = 0.0;  // camera timestamp of the image
  Sophus::SO3d R_GI;
  Eigen::Vector3d p_GI = Eigen::Vector3d::Zero();
  Sophus::SO3d R_GI_fej;
  Eigen::Vector3d p_GI_fej = Eigen::Vector3d::Zero();
};

// Error-state layout (plan, Stage 3 "State"), with true = estimate (+) error:
//
//   [0, 15)        IMU: dtheta (R_GI = R_hat Exp(dtheta)), dp, dv, db_g, db_a
//   [15, 21)       extrinsic, if estimated: dtheta_IC (R_IC = R_hat Exp), dp_IC
//   next 1         time offset t_d, if estimated
//   6 per clone    dtheta, dp of each clone, oldest first
//
// The IMU's FEJ copy is its estimate before the last update (the propagated
// one), which is where the next propagation's transition is evaluated.
class FilterState {
 public:
  FilterState(const ImuInitState& init, const Sophus::SO3d& R_IC, const Eigen::Vector3d& p_IC, double t_d,
              const StateOptions& options, const Eigen::Matrix<double, 7, 7>& calib_cov);

  const StateOptions& options() const { return options_; }

  int dim() const { return static_cast<int>(P.rows()); }
  int extrinsicIndex() const { return options_.estimate_extrinsic ? imu_err::kDim : -1; }
  int timeOffsetIndex() const;
  int cloneIndex(size_t i) const { return clones_start_ + 6 * static_cast<int>(i); }
  const std::deque<Clone>& clones() const { return clones_; }

  // Applies an error-state correction of size dim() to every estimate.
  void inject(const Eigen::VectorXd& dx);

  // Clones the IMU pose at camera time t_cam (the state must have been
  // propagated to t_cam + t_d) and augments P. omega is the bias-corrected
  // angular velocity at that time: the time-offset column of the clone
  // Jacobian is [omega; v].
  void addClone(double t_cam, const Eigen::Vector3d& omega);
  // Removes clone i and its rows and columns of P.
  void marginalizeClone(size_t i);

  // Symmetrizes P (after each update).
  void symmetrize() { P = 0.5 * (P + P.transpose()).eval(); }

  double t = 0.0;  // IMU state time (IMU clock)
  NavState nav;
  NavState nav_fej;
  /// IMU error transition of the last propagation, as applied to P (read by the observability check).
  Matrix15d Phi_last = Matrix15d::Identity();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
  Sophus::SO3d R_IC;
  Eigen::Vector3d p_IC = Eigen::Vector3d::Zero();
  double t_d = 0.0;
  Eigen::MatrixXd P;

 private:
  StateOptions options_;
  int clones_start_ = imu_err::kDim;
  std::deque<Clone> clones_;
};

}  // namespace msckf::filter
