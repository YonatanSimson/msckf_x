#include "msckf/camera/pinhole_radtan.h"

#include <Eigen/LU>

#include "monotonic_limit.h"

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
constexpr int kMaxIterations = 30;
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

void PinholeRadtan::updateMaxRadius() {
  // The valid disk is where the 2D distortion map stays a local diffeomorphism:
  // det(d distorted / d undistorted) > 0 along every azimuth. Without tangential
  // terms this reduces to radial monotonicity, 1 + 3k1 r^2 + 5k2 r^4 + 7k3 r^6 > 0;
  // with them, the fold moves with azimuth, so take the minimum over a ring.
  constexpr int kAzimuths = 360;
  r_max_ = kRadiusCap;
  for (int i = 0; i < kAzimuths; ++i) {
    const double phi = 2.0 * M_PI * i / kAzimuths;
    const Eigen::Vector2d dir(std::cos(phi), std::sin(phi));
    const double limit = internal::monotonicLimit(
        [&](double r) {
          Eigen::Matrix2d J;
          distort(r * dir, &J);
          return J.determinant();
        },
        r_max_);
    r_max_ = std::min(r_max_, limit);
  }
  // Margin so unprojection near the fold stays well conditioned.
  r_max_ *= 0.99;
}

Eigen::Vector2d PinholeRadtan::distort(const Eigen::Vector2d& xn, Eigen::Matrix2d* J) const {
  const double k1 = params_[4], k2 = params_[5], p1 = params_[6], p2 = params_[7], k3 = params_[8];
  const double x = xn.x(), y = xn.y();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
  Eigen::Vector2d xd(x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                     y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
  if (J) {
    const double drad = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2);  // d(radial)/d(r2)
    (*J)(0, 0) = radial + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x;
    (*J)(0, 1) = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y;
    (*J)(1, 0) = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y;
    (*J)(1, 1) = radial + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x;
  }
  return xd;
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

  Eigen::Matrix2d J_dist;
  const Eigen::Vector2d xd = distort(xn, J_point ? &J_dist : nullptr);
  uv << fx * xd.x() + cx, fy * xd.y() + cy;

  if (J_point) {
    Matrix23d J_norm;
    J_norm << inv_z, 0.0, -xn.x() * inv_z,
              0.0, inv_z, -xn.y() * inv_z;
    *J_point = Eigen::DiagonalMatrix<double, 2>(fx, fy) * J_dist * J_norm;
  }
  if (J_params) {
    const double x = xn.x(), y = xn.y();
    const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
    J_params->setZero(2, kNumParams);
    (*J_params)(0, 0) = xd.x();
    (*J_params)(1, 1) = xd.y();
    (*J_params)(0, 2) = 1.0;
    (*J_params)(1, 3) = 1.0;
    (*J_params)(0, 4) = fx * x * r2;
    (*J_params)(1, 4) = fy * y * r2;
    (*J_params)(0, 5) = fx * x * r4;
    (*J_params)(1, 5) = fy * y * r4;
    (*J_params)(0, 6) = fx * 2.0 * x * y;
    (*J_params)(1, 6) = fy * (r2 + 2.0 * y * y);
    (*J_params)(0, 7) = fx * (r2 + 2.0 * x * x);
    (*J_params)(1, 7) = fy * 2.0 * x * y;
    (*J_params)(0, 8) = fx * x * r6;
    (*J_params)(1, 8) = fy * y * r6;
  }
  return true;
}

bool PinholeRadtan::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const {
  const double fx = params_[0], fy = params_[1], cx = params_[2], cy = params_[3];
  const Eigen::Vector2d xd((uv.x() - cx) / fx, (uv.y() - cy) / fy);

  // Gauss-Newton (here Newton, the system is square) on distort(xn) = xd.
  // Stops on a vanishing step or a residual at machine precision, not on a loose
  // residual tolerance: near r_max the distortion is nearly flat, so a small
  // residual can still leave a large error in xn.
  Eigen::Vector2d xn = xd;
  bool converged = false;
  for (int i = 0; i < kMaxIterations; ++i) {
    Eigen::Matrix2d J;
    const Eigen::Vector2d err = distort(xn, &J) - xd;
    if (err.norm() <= 4.0 * std::numeric_limits<double>::epsilon() * (1.0 + xd.norm())) {
      converged = true;
      break;
    }
    Eigen::Vector2d step = J.inverse() * err;
    // Backtrack so the iterate never jumps past the fold onto another branch.
    for (int k = 0; k < 30 && (xn - step).norm() > r_max_; ++k) step *= 0.5;
    xn -= step;
    if (!xn.allFinite() || xn.norm() > 2.0 * kRadiusCap) return false;
    if (step.norm() < 1e-15 * (1.0 + xn.norm())) {
      converged = true;
      break;
    }
  }
  if (!converged || xn.norm() > r_max_ || (distort(xn) - xd).norm() > 1e-10) return false;
  bearing = Eigen::Vector3d(xn.x(), xn.y(), 1.0).normalized();
  return true;
}

}  // namespace msckf
