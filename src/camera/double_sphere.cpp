#include "msckf/camera/double_sphere.h"

#include <cmath>
#include <stdexcept>

namespace msckf {

DoubleSphere::DoubleSphere(int width, int height, const Eigen::VectorXd& params)
    : CameraModel(width, height) {
  setParams(params);
}

void DoubleSphere::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("DoubleSphere: expected 6 params");
  params_ = params;
}

bool DoubleSphere::validPoint(const Eigen::Vector3d& p_c) const {
  const double xi = params_[4], alpha = params_[5];
  const double d1 = p_c.norm();
  if (d1 == 0.0) return false;
  // Validity region from the paper: z > -w2 d1.
  const double w1 = alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
  const double w2 = (w1 + xi) / std::sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  return p_c.z() > -w2 * d1;
}

bool DoubleSphere::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                           Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double xi = params_[4], alpha = params_[5];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();

  const double d1 = p_c.norm();
  const double k = xi * d1 + z;
  const double d2 = std::sqrt(x * x + y * y + k * k);
  const double den = alpha * d2 + (1.0 - alpha) * k;
  if (den <= 0.0) return false;
  const double inv_den = 1.0 / den;
  uv << fx * x * inv_den + cx, fy * y * inv_den + cy;

  if (J_point) {
    const Eigen::Vector3d dk_dp = xi * p_c / d1 + Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d dd2_dp = (Eigen::Vector3d(x, y, 0.0) + k * dk_dp) / d2;
    const Eigen::Vector3d dden_dp = alpha * dd2_dp + (1.0 - alpha) * dk_dp;
    J_point->row(0) = fx * inv_den * (Eigen::RowVector3d::UnitX() - x * inv_den * dden_dp.transpose());
    J_point->row(1) = fy * inv_den * (Eigen::RowVector3d::UnitY() - y * inv_den * dden_dp.transpose());
  }
  if (J_params) {
    const double dden_dxi = (alpha * k / d2 + (1.0 - alpha)) * d1;
    const double dden_dalpha = d2 - k;
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = x * inv_den;
    (*J_params)(1, 1) = y * inv_den;
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    (*J_params)(0, 4) = -fx * x * inv_den * inv_den * dden_dxi;
    (*J_params)(1, 4) = -fy * y * inv_den * inv_den * dden_dxi;
    (*J_params)(0, 5) = -fx * x * inv_den * inv_den * dden_dalpha;
    (*J_params)(1, 5) = -fy * y * inv_den * inv_den * dden_dalpha;
  }
  return true;
}

bool DoubleSphere::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double xi = params_[4], alpha = params_[5];
  const double mx = (uv.x() - cx) / fx, my = (uv.y() - cy) / fy;
  const double r2 = mx * mx + my * my;

  const double s = 1.0 - (2.0 * alpha - 1.0) * r2;
  if (s < 0.0) return false;  // outside the image of the model (alpha > 0.5)
  const double mz = (1.0 - alpha * alpha * r2) / (alpha * std::sqrt(s) + 1.0 - alpha);
  const double q = mz * mz + (1.0 - xi * xi) * r2;
  if (q < 0.0) return false;
  const double factor = (mz * xi + std::sqrt(q)) / (mz * mz + r2);

  const Eigen::Vector3d p(factor * mx, factor * my, factor * mz - xi);
  if (!validPoint(p)) return false;
  bearing = p.normalized();
  return true;
}

}  // namespace msckf
