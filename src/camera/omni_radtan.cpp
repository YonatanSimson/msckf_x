#include "msckf/camera/omni_radtan.h"

#include "radtan_distortion.h"

#include <cmath>
#include <stdexcept>

namespace msckf {

namespace {
// Cap on the normalized-plane radius when xi <= 1, where the plane is unbounded
// as z approaches -xi |p|. Far outside any real image for practical focal lengths.
constexpr double kRadiusCap = 20.0;
}  // namespace

OmniRadtan::OmniRadtan(int width, int height, const Eigen::VectorXd& params)
    : CameraModel(width, height) {
  setParams(params);
}

void OmniRadtan::setParams(const Eigen::VectorXd& params) {
  if (params.size() != kNumParams) throw std::invalid_argument("OmniRadtan: expected 9 params");
  if (!(params[4] > 0.0)) throw std::invalid_argument("OmniRadtan: xi must be positive");
  params_ = params;
  updateMaxRadius();
}

internal::RadtanDistortion OmniRadtan::distortion() const {
  return {params_[5], params_[6], params_[7], params_[8], 0.0};
}

void OmniRadtan::updateMaxRadius() {
  const double xi = params_[4];
  // For xi > 1, points with z < -|p| / xi fold back onto the plane; their
  // boundary maps to the circle |m| = 1 / sqrt(xi^2 - 1).
  const double cap = xi > 1.0 ? 1.0 / std::sqrt(xi * xi - 1.0) : kRadiusCap;
  r_max_ = distortion().maxRadius(cap);
}

bool OmniRadtan::validPoint(const Eigen::Vector3d& p_c) const {
  const double xi = params_[4];
  const double d = p_c.norm();
  const double w = xi <= 1.0 ? xi : 1.0 / xi;
  if (d <= 0.0 || p_c.z() <= -w * d) return false;
  const double D = p_c.z() + xi * d;
  return D > 0.0 && p_c.head<2>().norm() <= r_max_ * D;
}

bool OmniRadtan::project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point,
                         Matrix2Xd* J_params) const {
  if (!validPoint(p_c)) return false;
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3], xi = params_[4];
  const double d = p_c.norm();
  const double D = p_c.z() + xi * d;
  const Eigen::Vector2d m = p_c.head<2>() / D;

  const internal::RadtanDistortion dist = distortion();
  Eigen::Matrix2d J_dist;
  const bool need_J = J_point || J_params;
  const Eigen::Vector2d md = dist.distort(m, need_J ? &J_dist : nullptr);
  uv << fx * md.x() + cx, fy * md.y() + cy;

  const Eigen::DiagonalMatrix<double, 2> F(fx, fy);
  if (J_point) {
    // dm/dp = ([I 0] - m dD/dp) / D, with dD/dp = e_z + xi p / d.
    Eigen::RowVector3d dD = xi * p_c.transpose() / d;
    dD(2) += 1.0;
    Matrix23d J_m;
    J_m << 1.0, 0.0, 0.0,
           0.0, 1.0, 0.0;
    J_m = (J_m - m * dD) / D;
    *J_point = F * J_dist * J_m;
  }
  if (J_params) {
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = md.x();
    (*J_params)(1, 1) = md.y();
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    // dm/dxi = -m d / D.
    J_params->col(4) = F * J_dist * (-m * d / D);
    J_params->block<2, 4>(0, 5) = F * dist.paramJacobian(m).leftCols<4>();
  }
  return true;
}

bool OmniRadtan::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3], xi = params_[4];
  const Eigen::Vector2d md((uv.x() - cx) / fx, (uv.y() - cy) / fy);
  Eigen::Vector2d m;
  if (!distortion().undistort(md, r_max_, &m)) return false;

  // Lift m back onto the unit sphere: the ray from (0, 0, -xi) through (m, 1)
  // meets the sphere at factor * (m, 1) - (0, 0, xi).
  const double rho2 = m.squaredNorm();
  const double disc = 1.0 + (1.0 - xi * xi) * rho2;
  if (disc < 0.0) return false;
  const double factor = (xi + std::sqrt(disc)) / (1.0 + rho2);
  bearing = Eigen::Vector3d(factor * m.x(), factor * m.y(), factor - xi);
  const double n = bearing.norm();
  if (!(n > 0.0)) return false;
  bearing /= n;
  return validPoint(bearing);
}

}  // namespace msckf
