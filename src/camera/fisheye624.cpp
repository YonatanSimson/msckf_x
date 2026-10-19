#include "msckf/camera/fisheye624.h"

#include "monotonic_limit.h"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msckf {

namespace {
// The model works on (x, y) / z, so it stops short of 90 degrees.
constexpr double kThetaCap = 0.5 * M_PI - 1e-3;
constexpr int kMaxIterations = 30;
constexpr int kNumRadial = 6;
constexpr int kFirstRadial = 4, kFirstTangential = 10, kFirstThinPrism = 12;
}  // namespace

Fisheye624::Fisheye624(int width, int height, const Eigen::VectorXd& params) : CameraModel(width, height) {
  setParams(params);
}

void Fisheye624::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("Fisheye624: expected 16 params");
  params_ = params;
  updateMaxTheta();
}

double Fisheye624::radial(double theta, double* d_dtheta) const {
  const double t2 = theta * theta;
  double d = 1.0, dd = 1.0, t2i = t2;
  for (int i = 0; i < kNumRadial; ++i) {
    const double k = params_[kFirstRadial + i];
    d += k * t2i;
    dd += (2.0 * i + 3.0) * k * t2i;
    t2i *= t2;
  }
  if (d_dtheta) *d_dtheta = dd;
  return theta * d;
}

void Fisheye624::updateMaxTheta() {
  theta_max_ = internal::monotonicLimit(
      [&](double t) {
        double dd;
        radial(t, &dd);
        return dd;
      },
      kThetaCap);
}

Eigen::Vector2d Fisheye624::distort(const Eigen::Vector2d& w, Eigen::Matrix2d* J_w) const {
  const Eigen::Vector2d p = params_.segment<2>(kFirstTangential);
  const Eigen::Vector4d s = params_.segment<4>(kFirstThinPrism);
  const double rho2 = w.squaredNorm();
  const Eigen::Vector2d prism(s[0] * rho2 + s[1] * rho2 * rho2, s[2] * rho2 + s[3] * rho2 * rho2);
  const Eigen::Vector2d w_d = w + 2.0 * w.dot(p) * w + rho2 * p + prism;
  if (J_w) {
    const Eigen::Vector2d dprism_drho2(s[0] + 2.0 * s[1] * rho2, s[2] + 2.0 * s[3] * rho2);
    *J_w = Eigen::Matrix2d::Identity() + 2.0 * (w * p.transpose() + w.dot(p) * Eigen::Matrix2d::Identity() +
                                               p * w.transpose() + dprism_drho2 * w.transpose());
  }
  return w_d;
}

bool Fisheye624::undistort(const Eigen::Vector2d& w_d, Eigen::Vector2d& w) const {
  // Newton on distort(w) = w_d; the distortion is a small perturbation of the identity.
  w = w_d;
  for (int i = 0; i < kMaxIterations; ++i) {
    Eigen::Matrix2d J;
    const Eigen::Vector2d err = distort(w, &J) - w_d;
    if (err.norm() < 1e-14) return true;
    w -= J.inverse() * err;
  }
  return false;
}

bool Fisheye624::solveTheta(double rd, double& theta) const {
  // Newton on d(theta) = rd; d is monotonic on [0, theta_max].
  theta = std::min(rd, theta_max_);
  for (int i = 0; i < kMaxIterations; ++i) {
    double dd;
    const double err = radial(theta, &dd) - rd;
    if (std::abs(err) < 1e-14) return true;
    theta = std::clamp(theta - err / dd, 0.0, theta_max_);
  }
  return false;
}

bool Fisheye624::validPoint(const Eigen::Vector3d& p_c) const {
  const double r = p_c.head<2>().norm();
  if (r == 0.0 && p_c.z() <= 0.0) return false;
  return std::atan2(r, p_c.z()) <= theta_max_;
}

void Fisheye624::paramJacobian(const Eigen::Vector2d& w, const Eigen::Vector2d& w_d, const Eigen::Vector2d& dw_dk,
                               const Eigen::Matrix2d& J_w, double theta2, Matrix2Xd& J) const {
  const Eigen::DiagonalMatrix<double, 2> F(params_[0], params_[1]);
  const double rho2 = w.squaredNorm();
  J.setZero(2, kNumParams);
  J(0, 0) = w_d.x();
  J(1, 1) = w_d.y();
  J(0, 2) = 1.0;
  J(1, 3) = 1.0;
  // d(w)/dk_i = theta^(2i+2) dw_dk, with dw_dk = theta (a, b) / |(a, b)|.
  Eigen::Vector2d col = F * (J_w * dw_dk) * theta2;
  for (int i = 0; i < kNumRadial; ++i) {
    J.col(kFirstRadial + i) = col;
    col *= theta2;
  }
  J.block<2, 2>(0, kFirstTangential) = F * (2.0 * w * w.transpose() + rho2 * Eigen::Matrix2d::Identity());
  J(0, kFirstThinPrism) = params_[0] * rho2;
  J(0, kFirstThinPrism + 1) = params_[0] * rho2 * rho2;
  J(1, kFirstThinPrism + 2) = params_[1] * rho2;
  J(1, kFirstThinPrism + 3) = params_[1] * rho2 * rho2;
}

bool Fisheye624::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                         Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();
  const double r2 = x * x + y * y;
  const double r = std::sqrt(r2);

  // w = m (x, y) with m = d(theta) / r; near the axis m -> 1 / z (a pinhole).
  const bool on_axis = r2 < 1e-16 * z * z;
  const double theta = on_axis ? 0.0 : std::atan2(r, z);
  double dd_dtheta = 1.0;
  const double m = on_axis ? 1.0 / z : radial(theta, &dd_dtheta) / r;
  const Eigen::Vector2d w(m * x, m * y);
  Eigen::Matrix2d J_w;
  const Eigen::Vector2d w_d = distort(w, (J_point || J_params) ? &J_w : nullptr);
  const Eigen::DiagonalMatrix<double, 2> F(params_[0], params_[1]);
  uv = F * w_d + params_.segment<2>(2);

  if (J_point) {
    Matrix23d dw_dp;
    if (on_axis) {
      const double inv_z = 1.0 / z;
      dw_dp << inv_z, 0.0, -x * inv_z * inv_z, 0.0, inv_z, -y * inv_z * inv_z;
    } else {
      const double rho2 = r2 + z * z;
      const Eigen::Vector3d dtheta_dp(z * x / (r * rho2), z * y / (r * rho2), -r / rho2);
      const Eigen::Vector3d dr_dp(x / r, y / r, 0.0);
      const Eigen::Vector3d dm_dp = (dd_dtheta * dtheta_dp - m * dr_dp) / r;
      dw_dp.row(0) = x * dm_dp.transpose() + Eigen::RowVector3d(m, 0.0, 0.0);
      dw_dp.row(1) = y * dm_dp.transpose() + Eigen::RowVector3d(0.0, m, 0.0);
    }
    *J_point = F * J_w * dw_dp;
  }
  if (J_params) {
    const Eigen::Vector2d dw_dk = on_axis ? Eigen::Vector2d::Zero() : Eigen::Vector2d(theta * x / r, theta * y / r);
    paramJacobian(w, w_d, dw_dk, J_w, theta * theta, *J_params);
  }
  return true;
}

bool Fisheye624::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const Eigen::Vector2d w_d((uv.x() - params_[2]) / params_[0], (uv.y() - params_[3]) / params_[1]);
  Eigen::Vector2d w;
  if (!undistort(w_d, w)) return false;
  const double rd = w.norm();
  if (rd < 1e-15) {
    bearing = Eigen::Vector3d::UnitZ();
    return true;
  }
  double theta;
  if (rd > radial(theta_max_) || !solveTheta(rd, theta)) return false;
  const double s = std::sin(theta) / rd;
  bearing << s * w.x(), s * w.y(), std::cos(theta);
  return true;
}

}  // namespace msckf
