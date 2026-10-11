#pragma once

#include "msckf/filter/msckf.h"
#include "msckf/sim/simulator.h"

#include <sophus/se3.hpp>

#include <cstdint>
#include <vector>

namespace msckf::filter {

// How a simulated filter run starts: at the true state at the first frame at
// or after t_start, minus an error drawn from the initial covariance (so the
// NEES of a Monte Carlo set is meaningful). The same for the calibration.
struct SimRunConfig {
  double t_start = 1.0;
  double sigma_theta = 0.5 * M_PI / 180.0;  // rad, each axis
  double sigma_p = 0.01;                    // m
  double sigma_v = 0.05;                    // m/s
  double sigma_bg = 2e-3;                   // rad/s
  double sigma_ba = 2e-2;                   // m/s^2
  double sigma_ext_theta = 0.5 * M_PI / 180.0;  // rad
  double sigma_ext_p = 0.01;                    // m
  double sigma_td = 0.005;                      // s
  bool perturb = true;  // draw the initial errors; false starts at the truth
  uint64_t seed = 1;
};

// Filter state against the truth after each image.
struct SimEpoch {
  double t = 0.0;  // camera time
  Sophus::SE3d T_est, T_true;     // IMU pose at the image time
  Eigen::Matrix<double, 6, 6> cov_pose = Eigen::Matrix<double, 6, 6>::Zero();  // [dtheta, dp]
  Eigen::Vector3d v_est = Eigen::Vector3d::Zero(), v_true = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov_v = Eigen::Matrix3d::Zero();
  double nees_bg = 0.0, nees_ba = 0.0;  // gyro / accelerometer bias, 3 dof each
  Eigen::Vector3d ext_theta_err = Eigen::Vector3d::Zero();  // Log(R_IC_est^T R_IC_true)
  Eigen::Vector3d ext_p_err = Eigen::Vector3d::Zero();
  double td_err = 0.0;
  int observations = 0;  // in the image
  UpdateStats update;
  double time_ms = 0.0;
};

// camera: the simulator's camera model.
std::vector<SimEpoch> runFilterOnSimulation(const Simulator& sim, const CameraModelPtr& camera, const SimData& data,
                                            const MsckfOptions& options, const SimRunConfig& config);

}  // namespace msckf::filter
