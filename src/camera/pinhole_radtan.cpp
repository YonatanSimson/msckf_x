#include "msckf/camera/pinhole_radtan.h"

#include "radtan_distortion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace msckf {

namespace {
constexpr double kMinZ = 1e-6;
// Distortion is only modeled up to this normalized radius (~87 degrees) even if
// it stays monotonic further out; beyond it the pinhole model is meaningless.
constexpr double kRadiusCap = 20.0;
}  // namespace

PinholeRadtan::PinholeRadtan(int width, int height, const Eigen::VectorXd& params)
    : CameraModel(width, height) {
  setParams(params);
}

void PinholeRadtan::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("PinholeRadtan: expected 9 params");
  params_ = params;
  updateMaxRadius();
}

void PinholeRadtan::updateMaxRadius() { r_max_ = distortion().maxRadius(kRadiusCap); }

internal::RadtanDistortion PinholeRadtan::distortion() const {
  return {params_[4], params_[5], params_[6], params_[7], params_[8]};
}

bool PinholeRadtan::validPoint(const Eigen::Vector3d& p_c) const {
  if (p_c.z() < kMinZ) return false;
  return p_c.head<2>().norm() <= r_max_ * p_c.z();
}

bool PinholeRadtan::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                            Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const double inv_z = 1.0 / p_c.z();
  const Eigen::Vector2d xn = p_c.head<2>() * inv_z;

  const internal::RadtanDistortion dist = distortion();
  Eigen::Matrix2d J_dist;
  const Eigen::Vector2d xd = dist.distort(xn, J_point ? &J_dist : nullptr);
  uv << fx * xd.x() + cx, fy * xd.y() + cy;

  if (J_point) {
    Matrix23d J_norm;
    J_norm << inv_z, 0.0, -xn.x() * inv_z,
              0.0, inv_z, -xn.y() * inv_z;
    *J_point = Eigen::DiagonalMatrix<double, 2>(fx, fy) * J_dist * J_norm;
  }
  if (J_params) {
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = xd.x();
    (*J_params)(1, 1) = xd.y();
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    J_params->block<2, 5>(0, 4) =
        Eigen::DiagonalMatrix<double, 2>(fx, fy) * dist.paramJacobian(xn);
  }
  return true;
}

bool PinholeRadtan::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const Eigen::Vector2d xd((uv.x() - cx) / fx, (uv.y() - cy) / fy);

  Eigen::Vector2d xn;
  if (!distortion().undistort(xd, r_max_, &xn)) return false;
  bearing = Eigen::Vector3d(xn.x(), xn.y(), 1.0).normalized();
  return true;
}

}  // namespace msckf
