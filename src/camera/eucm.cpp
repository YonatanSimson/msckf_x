#include "msckf/camera/eucm.h"

#include <cmath>
#include <stdexcept>

namespace msckf {

Eucm::Eucm(int width, int height, const Eigen::VectorXd& params) : CameraModel(width, height) {
  setParams(params);
}

void Eucm::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("Eucm: expected 6 params");
  params_ = params;
}

bool Eucm::validPoint(const Eigen::Vector3d& p_c) const {
  const double alpha = params_[4], beta = params_[5];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();
  const double d = std::sqrt(beta * (x * x + y * y) + z * z);
  if (d == 0.0) return false;
  const double w = alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
  return z > -w * d;
}

bool Eucm::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                   Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double alpha = params_[4], beta = params_[5];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();

  const double rho2 = x * x + y * y;
  const double d = std::sqrt(beta * rho2 + z * z);
  const double den = alpha * d + (1.0 - alpha) * z;
  if (den <= 0.0) return false;
  const double inv_den = 1.0 / den;
  uv << fx * x * inv_den + cx, fy * y * inv_den + cy;

  if (J_point) {
    const Eigen::Vector3d dd_dp(beta * x / d, beta * y / d, z / d);
    const Eigen::Vector3d dden_dp = alpha * dd_dp + (1.0 - alpha) * Eigen::Vector3d::UnitZ();
    J_point->row(0) = fx * inv_den * (Eigen::RowVector3d::UnitX() - x * inv_den * dden_dp.transpose());
    J_point->row(1) = fy * inv_den * (Eigen::RowVector3d::UnitY() - y * inv_den * dden_dp.transpose());
  }
  if (J_params) {
    const double dden_dalpha = d - z;
    const double dden_dbeta = alpha * rho2 / (2.0 * d);
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = x * inv_den;
    (*J_params)(1, 1) = y * inv_den;
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    (*J_params)(0, 4) = -fx * x * inv_den * inv_den * dden_dalpha;
    (*J_params)(1, 4) = -fy * y * inv_den * inv_den * dden_dalpha;
    (*J_params)(0, 5) = -fx * x * inv_den * inv_den * dden_dbeta;
    (*J_params)(1, 5) = -fy * y * inv_den * inv_den * dden_dbeta;
  }
  return true;
}

bool Eucm::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double alpha = params_[4], beta = params_[5];
  const double mx = (uv.x() - cx) / fx, my = (uv.y() - cy) / fy;
  const double r2 = mx * mx + my * my;

  const double s = 1.0 - (2.0 * alpha - 1.0) * beta * r2;
  if (s < 0.0) return false;  // outside the image of the model (alpha > 0.5)
  const double mz = (1.0 - beta * alpha * alpha * r2) / (alpha * std::sqrt(s) + 1.0 - alpha);

  const Eigen::Vector3d p(mx, my, mz);
  if (!validPoint(p)) return false;
  bearing = p.normalized();
  return true;
}

}  // namespace msckf
