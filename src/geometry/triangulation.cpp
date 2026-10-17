#include "msckf/geometry/triangulation.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace msckf {

Matrix32d tangentBasis(const Eigen::Vector3d& b) {
  const Eigen::Vector3d a = std::abs(b.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  Matrix32d B;
  B.col(0) = b.cross(a).normalized();
  B.col(1) = b.cross(B.col(0));
  return B;
}

double maxRayAngleDeg(const std::vector<Eigen::Vector3d>& directions) {
  double best = 0.0;
  for (size_t i = 0; i < directions.size(); ++i) {
    for (size_t j = i + 1; j < directions.size(); ++j) {
      const Eigen::Vector3d& a = directions[i];
      const Eigen::Vector3d& b = directions[j];
      best = std::max(best, std::atan2(a.cross(b).norm(), a.dot(b)));
    }
  }
  return best * 180.0 / M_PI;
}

namespace {

// Least squares on the perpendicular distances to the rays from the camera
// centres along the global directions d. False for rays too close to parallel.
bool midpoint(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& d, Eigen::Vector3d* X) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < d.size(); ++i) {
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d[i] * d[i].transpose();
    A += P;
    rhs += P * T_GC[i].translation();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.eigenvalues()(0) < 1e-9 * es.eigenvalues()(2)) return false;
  *X = A.ldlt().solve(rhs);
  return true;
}

// Anchor frame A: the anchor camera turned so that its z axis is the measured
// bearing b, so every ray, also past 90 deg off the optical axis, starts near
// alpha = beta = 0.
Sophus::SE3d anchorFrame(const Sophus::SE3d& T_GC, const Eigen::Vector3d& b) {
  Eigen::Matrix3d R;
  R << tangentBasis(b), b;
  return T_GC * Sophus::SE3d(Sophus::SO3d(Sophus::makeRotationMatrix(R)), Eigen::Vector3d::Zero());
}

// Gauss-Newton step on the tangent-plane bearing residuals for the anchored
// inverse depth x = (alpha, beta, rho), p_A = [alpha, beta, 1] / rho. With
// T_CA the camera poses relative to A, rho p_C = R_CA [alpha, beta, 1] + rho t_CA,
// which stays finite for a point at infinity (rho = 0).
Eigen::Vector3d inverseDepthStep(const std::vector<Sophus::SE3d>& T_CA, const std::vector<Eigen::Vector3d>& b,
                                 const Eigen::Vector3d& x) {
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
  const Eigen::Vector3d ray(x.x(), x.y(), 1.0);
  for (size_t i = 0; i < b.size(); ++i) {
    const Eigen::Matrix3d R = T_CA[i].so3().matrix();
    const Eigen::Vector3d h = R * ray + x.z() * T_CA[i].translation();
    const double r = h.norm();
    const Eigen::Vector3d n = h / r;
    Eigen::Matrix3d dh_dx;
    dh_dx << R.col(0), R.col(1), T_CA[i].translation();
    const Matrix32d B = tangentBasis(b[i]);
    const Eigen::Matrix<double, 2, 3> J = B.transpose() * (Eigen::Matrix3d::Identity() - n * n.transpose()) / r * dh_dx;
    H += J.transpose() * J;
    g += J.transpose() * (-B.transpose() * n);
  }
  return H.ldlt().solve(g);
}

}  // namespace

bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, Eigen::Vector3d* X,
                 double* parallax_deg) {
  std::vector<Eigen::Vector3d> d(b.size());
  for (size_t i = 0; i < b.size(); ++i) d[i] = T_GC[i].so3() * b[i];
  *parallax_deg = maxRayAngleDeg(d);
  Eigen::Vector3d X0;
  if (!midpoint(T_GC, d, &X0)) return false;
  const Sophus::SE3d T_GA = anchorFrame(T_GC.front(), b.front());
  const Eigen::Vector3d p_A = T_GA.inverse() * X0;
  if (p_A.z() <= 0.0) return false;
  Eigen::Vector3d x(p_A.x() / p_A.z(), p_A.y() / p_A.z(), 1.0 / p_A.z());
  std::vector<Sophus::SE3d> T_CA;
  for (const Sophus::SE3d& T : T_GC) T_CA.push_back(T.inverse() * T_GA);
  for (int it = 0; it < 5; ++it) {
    const Eigen::Vector3d dx = inverseDepthStep(T_CA, b, x);
    x += dx;
    if (dx.norm() < 1e-9 * (1.0 + x.norm())) break;
  }
  if (x.z() <= 0.0) return false;
  *X = T_GA * (Eigen::Vector3d(x.x(), x.y(), 1.0) / x.z());
  for (size_t i = 0; i < b.size(); ++i) {
    if ((T_GC[i].inverse() * *X).dot(b[i]) <= 0.0) return false;
  }
  return true;
}

}  // namespace msckf
