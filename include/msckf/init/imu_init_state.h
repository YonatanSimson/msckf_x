#pragma once

#include "msckf/imu/imu_types.h"

#include <Eigen/Core>

namespace msckf {

// IMU error-state layout shared by initialization and the filter (plan,
// Stage 3 "State"): R_GI = R_hat Exp(dtheta) with dtheta in the IMU frame,
// then position, velocity, gyro bias, accelerometer bias, true = est + error.
namespace imu_err {
constexpr int kTheta = 0;
constexpr int kP = 3;
constexpr int kV = 6;
constexpr int kBg = 9;
constexpr int kBa = 12;
constexpr int kDim = 15;
}  // namespace imu_err

using Matrix15d = Eigen::Matrix<double, imu_err::kDim, imu_err::kDim>;

// What an initializer hands to the filter: the IMU state at time t and the
// covariance of its 15-dim error.
struct ImuInitState {
  double t = 0.0;
  NavState nav;
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
  Matrix15d P = Matrix15d::Zero();
};

}  // namespace msckf
