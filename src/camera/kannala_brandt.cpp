#include "msckf/camera/kannala_brandt.h"

#include "monotonic_limit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msckf {

namespace {
// Stay clear of theta = pi, where the azimuth is undefined.
constexpr double kThetaCap = M_PI - 1e-3;
constexpr int kMaxIterations = 30;
}  // namespace

KannalaBrandt::KannalaBrandt(int width, int height, const Eigen::VectorXd& params)
    : CameraModel(width, height) {
  setParams(params);
}

void KannalaBrandt::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("KannalaBrandt: expected 8 params");
  params_ = params;
  updateMaxTheta();
}

double KannalaBrandt::poly(double theta, double* d_dtheta) const {
  const double k1 = params_[4], k2 = params_[5], k3 = params_[6], k4 = params_[7];
  const double t2 = theta * theta;
  if (d_dtheta) {
    *d_dtheta = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
  }
  return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
}

void KannalaBrandt::updateMaxTheta() {
  theta_max_ = internal::monotonicLimit(
      [&](double t) {
        double dd;
        poly(t, &dd);
        return dd;
      },
      kThetaCap);
}

bool KannalaBrandt::validPoint(const Eigen::Vector3d& p_c) const {
  const double r = p_c.head<2>().norm();
  if (r == 0.0 && p_c.z() <= 0.0) return false;
  return std::atan2(r, p_c.z()) <= theta_max_;
}

bool KannalaBrandt::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                            Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();
  const double r2 = x * x + y * y;
  const double r = std::sqrt(r2);

  // Near the optical axis d(theta)/r -> 1/z: the model is a pinhole to O(r^2).
  if (r2 < 1e-16 * z * z) {
    const double inv_z = 1.0 / z;
    uv << fx * x * inv_z + cx, fy * y * inv_z + cy;
    if (J_point) {
      *J_point << fx * inv_z, 0.0, -fx * x * inv_z * inv_z,
                  0.0, fy * inv_z, -fy * y * inv_z * inv_z;
    }
    if (J_params) {
      J_params->setZero(2, kNumParams);
      (*J_params)(0, 0) = x * inv_z;
      (*J_params)(1, 1) = y * inv_z;
      (*J_params)(0, 2) = 1.0;
      (*J_params)(1, 3) = 1.0;
    }
    return true;
  }

  const double theta = std::atan2(r, z);
  double dd_dtheta;
  const double d = poly(theta, &dd_dtheta);
  const double f = d / r;  // pixels per unit of (x, y), before focal scaling
  uv << fx * f * x + cx, fy * f * y + cy;

  if (J_point) {
    const double rho2 = r2 + z * z;
    const Eigen::Vector3d dtheta_dp(z * x / (r * rho2), z * y / (r * rho2), -r / rho2);
    const Eigen::Vector3d dr_dp(x / r, y / r, 0.0);
    // f = d(theta(p)) / r(p)
    const Eigen::Vector3d df_dp = (dd_dtheta * dtheta_dp - f * dr_dp) / r;
    J_point->row(0) = fx * (x * df_dp.transpose() + Eigen::RowVector3d(f, 0.0, 0.0));
    J_point->row(1) = fy * (y * df_dp.transpose() + Eigen::RowVector3d(0.0, f, 0.0));
  }
  if (J_params) {
    const double mx = x / r, my = y / r;
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = f * x;
    (*J_params)(1, 1) = f * y;
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    double theta_pow = theta * theta * theta;  // d(d)/dk_i = theta^(2i+1)
    for (int i = 0; i < 4; ++i) {
      (*J_params)(0, 4 + i) = fx * mx * theta_pow;
      (*J_params)(1, 4 + i) = fy * my * theta_pow;
      theta_pow *= theta * theta;
    }
  }
  return true;
}

bool KannalaBrandt::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double mx = (uv.x() - cx) / fx, my = (uv.y() - cy) / fy;
  const double rd = std::sqrt(mx * mx + my * my);
  if (rd < 1e-15) {
    bearing = Eigen::Vector3d::UnitZ();
    return true;
  }
  if (rd > poly(theta_max_)) return false;

  // Newton on d(theta) = rd; d is monotonic on [0, theta_max].
  double theta = std::min(rd, theta_max_);
  bool converged = false;
  for (int i = 0; i < kMaxIterations; ++i) {
    double dd;
    const double err = poly(theta, &dd) - rd;
    if (std::abs(err) < 1e-14) {
      converged = true;
      break;
    }
    theta = std::clamp(theta - err / dd, 0.0, theta_max_);
  }
  if (!converged) return false;

  const double s = std::sin(theta) / rd;
  bearing << s * mx, s * my, std::cos(theta);
  return true;
}

}  // namespace msckf
