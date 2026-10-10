#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace msckf {

// One raw IMU sample: gyro (rad/s) and specific force (m/s^2) in the IMU frame.
struct ImuSample {
  double t = 0.0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
};

// Continuous-time noise densities, as on a datasheet or from Allan variance.
struct ImuNoise {
  double gyro_noise = 1.7e-4;        // rad/s/sqrt(Hz)
  double acc_noise = 2.0e-3;         // m/s^2/sqrt(Hz)
  double gyro_bias_walk = 1.9e-5;    // rad/s^2/sqrt(Hz)
  double acc_bias_walk = 3.0e-3;     // m/s^3/sqrt(Hz)
};

// Nominal navigation state propagated by the IMU.
struct NavState {
  Sophus::SO3d R_GI;
  Eigen::Vector3d p_GI = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_GI = Eigen::Vector3d::Zero();
};

}  // namespace msckf
