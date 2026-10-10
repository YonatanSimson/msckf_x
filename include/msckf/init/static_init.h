#pragma once

#include "msckf/imu/imu_types.h"
#include "msckf/init/imu_init_state.h"

#include <optional>
#include <string>
#include <vector>

namespace msckf {

// Stage 1a: initialization from a stationary window (plan, "Static
// initialization"). The primary path for the cart, which almost always starts
// at rest.
struct StaticInitConfig {
  double window = 1.0;  // s, length of the stationary window ending at t_end

  // Stationarity. The accelerometer test alone passes a cart rolling at
  // constant velocity, so the tracker's median feature disparity over the
  // window is also required when it is available.
  double max_acc_std = 0.1;         // m/s^2, sqrt of the trace of the sample covariance
  double max_disparity_px = 1.0;    // median feature disparity across the window
  double max_gravity_error = 0.3;   // m/s^2, | |mean specific force - b_a| - g |

  ImuNoise noise;

  // Accelerometer bias prior (from a previous calibration). It cannot be
  // separated from tilt in a static window, so its uncertainty goes straight
  // into roll and pitch.
  Eigen::Vector3d acc_bias_prior = Eigen::Vector3d::Zero();
  double acc_bias_sigma = 0.05;  // m/s^2, per axis

  // Gauge and assumption sigmas. Yaw and position are unobservable and only
  // fix the gauge; velocity is zero by assumption.
  double yaw_sigma = 1e-3;       // rad
  double position_sigma = 1e-3;  // m
  double velocity_sigma = 1e-3;  // m/s
};

struct StationarityCheck {
  bool stationary = false;
  double acc_std = 0.0;             // m/s^2
  double disparity_px = -1.0;       // < 0 when not provided
  double gravity_error = 0.0;       // m/s^2, | |mean specific force - b_a| - g |
  int num_samples = 0;
  std::string reason;               // why the window is not stationary
};

// Uses the IMU samples in [t_end - window, t_end].
StationarityCheck checkStationary(const std::vector<ImuSample>& imu, double t_end,
                                  const StaticInitConfig& config,
                                  std::optional<double> median_disparity_px = std::nullopt);

// Initial state at t_end from a stationary window: roll and pitch from the mean
// specific force, yaw = 0 (the global x axis is the horizontal projection of
// the IMU x axis), p = 0, v = 0, b_g = mean gyro, b_a = prior. The covariance
// is the first-order propagation of the window's sensor noise and the
// accelerometer bias prior, including the tilt / b_a correlation. Returns
// nothing if the window is not stationary.
std::optional<ImuInitState> staticInitialize(const std::vector<ImuSample>& imu, double t_end,
                                             const StaticInitConfig& config,
                                             std::optional<double> median_disparity_px = std::nullopt,
                                             StationarityCheck* check = nullptr);

}  // namespace msckf
