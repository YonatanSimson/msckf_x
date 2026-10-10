#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

#include "monotonic_limit.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace msckf::internal {

// Brown-Conrady radial-tangential distortion on a normalized image plane,
// shared by PinholeRadtan and OmniRadtan:
//   xd = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
//   yd = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
struct RadtanDistortion {
  double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0;

  // J is d(distorted)/d(undistorted).
  Eigen::Vector2d distort(const Eigen::Vector2d& xn, Eigen::Matrix2d* J = nullptr) const {
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

  // d(distorted)/d(k1, k2, p1, p2, k3) at xn.
  Eigen::Matrix<double, 2, 5> paramJacobian(const Eigen::Vector2d& xn) const {
    const double x = xn.x(), y = xn.y();
    const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
    Eigen::Matrix<double, 2, 5> J;
    J << x * r2, x * r4, 2.0 * x * y, r2 + 2.0 * x * x, x * r6,
         y * r2, y * r4, r2 + 2.0 * y * y, 2.0 * x * y, y * r6;
    return J;
  }

  // Radius of the disk on which the 2D distortion map stays a local
  // diffeomorphism: det(d distorted / d undistorted) > 0 along every azimuth.
  // Without tangential terms this reduces to radial monotonicity,
  // 1 + 3k1 r^2 + 5k2 r^4 + 7k3 r^6 > 0; with them, the fold moves with
  // azimuth, so take the minimum over a ring. Includes a 1% margin so
  // undistortion near the fold stays well conditioned.
  double maxRadius(double cap) const {
    constexpr int kAzimuths = 360;
    double r_max = cap;
    for (int i = 0; i < kAzimuths; ++i) {
      const double phi = 2.0 * M_PI * i / kAzimuths;
      const Eigen::Vector2d dir(std::cos(phi), std::sin(phi));
      const double limit = monotonicLimit(
          [&](double r) {
            Eigen::Matrix2d J;
            distort(r * dir, &J);
            return J.determinant();
          },
          r_max);
      r_max = std::min(r_max, limit);
    }
    return 0.99 * r_max;
  }

  // Inverts distort() inside the disk |xn| <= r_max. Newton on the square
  // system, stopping on a vanishing step or a residual at machine precision,
  // not on a loose residual tolerance: near r_max the distortion is nearly
  // flat, so a small residual can still leave a large error in xn.
  bool undistort(const Eigen::Vector2d& xd, double r_max, Eigen::Vector2d* xn_out) const {
    constexpr int kMaxIterations = 30;
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
      for (int k = 0; k < 30 && (xn - step).norm() > r_max; ++k) step *= 0.5;
      xn -= step;
      if (!xn.allFinite() || xn.norm() > 2.0 * r_max + 1.0) return false;
      if (step.norm() < 1e-15 * (1.0 + xn.norm())) {
        converged = true;
        break;
      }
    }
    if (!converged || xn.norm() > r_max || (distort(xn) - xd).norm() > 1e-10) return false;
    *xn_out = xn;
    return true;
  }
};

}  // namespace msckf::internal
