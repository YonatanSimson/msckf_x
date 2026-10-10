#include "msckf/init/static_init.h"

#include "msckf/core/constants.h"

#include <cmath>
#include <sstream>

namespace msckf {

namespace {

struct WindowStats {
  int n = 0;
  double dt = 0.0;  // mean sample interval
  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
  double acc_std = 0.0;  // sqrt(trace of the sample covariance)
};

WindowStats windowStats(const std::vector<ImuSample>& imu, double t_end, double window) {
  constexpr double kEps = 1e-9;
  WindowStats w;
  double t_first = 0.0, t_last = 0.0;
  Eigen::Vector3d acc_sq = Eigen::Vector3d::Zero();
  for (const ImuSample& s : imu) {
    if (s.t < t_end - window - kEps || s.t > t_end + kEps) continue;
    if (w.n == 0) t_first = s.t;
    t_last = s.t;
    ++w.n;
    w.acc_mean += s.acc;
    w.gyro_mean += s.gyro;
    acc_sq += s.acc.cwiseProduct(s.acc);
  }
  if (w.n < 2) return w;
  w.acc_mean /= w.n;
  w.gyro_mean /= w.n;
  const Eigen::Vector3d var =
      (acc_sq - w.n * w.acc_mean.cwiseProduct(w.acc_mean)) / static_cast<double>(w.n - 1);
  w.acc_std = std::sqrt(std::max(0.0, var.sum()));
  w.dt = (t_last - t_first) / (w.n - 1);
  return w;
}

// Attitude with R_GI^T e_z = z (z: gravity's "up" in the IMU frame) and the
// global x axis along the horizontal projection of the IMU x axis (yaw = 0).
Sophus::SO3d attitudeFromUp(const Eigen::Vector3d& z) {
  Eigen::Vector3d x = Eigen::Vector3d::UnitX() - z.x() * z;
  if (x.norm() < 1e-3) x = Eigen::Vector3d::UnitY() - z.y() * z;  // IMU x is vertical
  x.normalize();
  const Eigen::Vector3d y = z.cross(x);
  Eigen::Matrix3d R_GI;  // rows: global axes expressed in the IMU frame
  R_GI.row(0) = x.transpose();
  R_GI.row(1) = y.transpose();
  R_GI.row(2) = z.transpose();
  return Sophus::SO3d(Sophus::makeRotationMatrix(R_GI));
}

}  // namespace

StationarityCheck checkStationary(const std::vector<ImuSample>& imu, double t_end,
                                  const StaticInitConfig& config,
                                  std::optional<double> median_disparity_px) {
  StationarityCheck c;
  const WindowStats w = windowStats(imu, t_end, config.window);
  c.num_samples = w.n;
  c.acc_std = w.acc_std;
  c.disparity_px = median_disparity_px.value_or(-1.0);
  if (w.n >= 2) {
    c.gravity_error = std::abs((w.acc_mean - config.acc_bias_prior).norm() - kGravity.norm());
  }
  std::ostringstream why;
  if (w.n < 10) {
    why << "only " << w.n << " IMU samples in the window";
  } else if (w.acc_std > config.max_acc_std) {
    why << "accelerometer std " << w.acc_std << " > " << config.max_acc_std << " m/s^2";
  } else if (median_disparity_px && *median_disparity_px > config.max_disparity_px) {
    why << "median disparity " << *median_disparity_px << " > " << config.max_disparity_px << " px";
  } else if (c.gravity_error > config.max_gravity_error) {
    why << "specific force norm off gravity by " << c.gravity_error << " m/s^2";
  } else {
    c.stationary = true;
  }
  c.reason = why.str();
  return c;
}

std::optional<ImuInitState> staticInitialize(const std::vector<ImuSample>& imu, double t_end,
                                             const StaticInitConfig& config,
                                             std::optional<double> median_disparity_px,
                                             StationarityCheck* check_out) {
  const StationarityCheck check = checkStationary(imu, t_end, config, median_disparity_px);
  if (check_out) *check_out = check;
  if (!check.stationary) return std::nullopt;
  const WindowStats w = windowStats(imu, t_end, config.window);

  // At rest a_m = R_GI^T (-g) + b_a + n_a, so f = mean(a_m) - b_a is gravity's
  // "up" in the IMU frame, scaled by |g|.
  const Eigen::Vector3d f = w.acc_mean - config.acc_bias_prior;
  const double f_norm = f.norm();
  const Eigen::Vector3d z = f / f_norm;

  ImuInitState s;
  s.t = t_end;
  s.nav.R_GI = attitudeFromUp(z);
  s.gyro_bias = w.gyro_mean;  // the earth's rotation (7e-5 rad/s) is ignored
  s.acc_bias = config.acc_bias_prior;

  // Errors are true - estimate. With z_true = Exp(-dtheta) z = z + [z]x dtheta
  // and z_true - z = (I - z z^T)(f_true - f) / |f|, the tilt part of dtheta is
  //   dtheta_tilt = [z]x (n_mean + db_a) / |f|,
  // where n_mean is the window-averaged accelerometer noise and db_a the bias
  // error. The component along z is yaw, the gauge.
  const double T = w.n * w.dt;  // window-averaged white noise has variance sigma^2 / T
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = Sophus::SO3d::hat(z);
  const Eigen::Matrix3d cov_acc_mean = (config.noise.acc_noise * config.noise.acc_noise / T) * I3;
  const Eigen::Matrix3d cov_ba = config.acc_bias_sigma * config.acc_bias_sigma * I3;

  using namespace imu_err;
  Matrix15d& P = s.P;
  P.setZero();
  P.block<3, 3>(kTheta, kTheta) = S * (cov_acc_mean + cov_ba) * S.transpose() / (f_norm * f_norm) +
                                  config.yaw_sigma * config.yaw_sigma * z * z.transpose();
  P.block<3, 3>(kTheta, kBa) = S * cov_ba / f_norm;
  P.block<3, 3>(kBa, kTheta) = P.block<3, 3>(kTheta, kBa).transpose();
  // The window constrains the window-averaged biases, but the state is at
  // t_end: a random walk with density s ends up off its window mean by a
  // variance of s^2 T / 3. For the accelerometer this is comparable to the
  // averaged white noise, and tilt and b_a are otherwise almost perfectly
  // correlated, so leaving it out makes the joint covariance overconfident.
  const double walk_ba = config.noise.acc_bias_walk * config.noise.acc_bias_walk * T / 3.0;
  const double walk_bg = config.noise.gyro_bias_walk * config.noise.gyro_bias_walk * T / 3.0;
  P.block<3, 3>(kBa, kBa) = cov_ba + walk_ba * I3;
  P.block<3, 3>(kBg, kBg) =
      (config.noise.gyro_noise * config.noise.gyro_noise / T + walk_bg) * I3;
  P.block<3, 3>(kP, kP) = config.position_sigma * config.position_sigma * I3;
  P.block<3, 3>(kV, kV) = config.velocity_sigma * config.velocity_sigma * I3;
  return s;
}

}  // namespace msckf
