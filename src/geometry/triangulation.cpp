#include "msckf/geometry/triangulation.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

Eigen::Vector3d AnchoredPoint::scaledFrom(const Eigen::Vector3d& c) const {
  return T_GA.so3() * Eigen::Vector3d(x.x(), x.y(), 1.0) + x.z() * (T_GA.translation() - c);
}

Eigen::Vector3d AnchoredPoint::point() const {
  if (x.z() <= 0.0) throw std::domain_error("AnchoredPoint::point: point at infinity (rho <= 0)");
  return T_GA * (Eigen::Vector3d(x.x(), x.y(), 1.0) / x.z());
}

AnchoredPoint AnchoredPoint::fromPoint(const Sophus::SE3d& T_GA, const Eigen::Vector3d& p_G) {
  const Eigen::Vector3d p_A = T_GA.inverse() * p_G;
  if (p_A.z() <= 0.0) throw std::domain_error("AnchoredPoint::fromPoint: point behind the anchor");
  return {T_GA, Eigen::Vector3d(p_A.x() / p_A.z(), p_A.y() / p_A.z(), 1.0 / p_A.z())};
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

// Start: the midpoint when the rays meet in front of the anchor, else the
// anchor's bearing at infinity (parallel rays, or no baseline).
Eigen::Vector3d initialInverseDepth(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& d,
                                    const Sophus::SE3d& T_GA) {
  Eigen::Vector3d X;
  if (!midpoint(T_GC, d, &X)) return Eigen::Vector3d::Zero();
  const Eigen::Vector3d p_A = T_GA.inverse() * X;
  if (p_A.z() <= 0.0) return Eigen::Vector3d::Zero();
  return {p_A.x() / p_A.z(), p_A.y() / p_A.z(), 1.0 / p_A.z()};
}

struct NormalEquations {
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
};

// Gauss-Newton normal equations of the tangent-plane bearing residuals for the
// anchored inverse depth x = (alpha, beta, rho), p_A = [alpha, beta, 1] / rho.
// With T_CA the camera poses relative to A, rho p_C = R_CA [alpha, beta, 1] +
// rho t_CA, which stays finite for a point at infinity (rho = 0).
NormalEquations normalEquations(const std::vector<Sophus::SE3d>& T_CA, const std::vector<Eigen::Vector3d>& b,
                                const Eigen::Vector3d& x) {
  NormalEquations ne;
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
    ne.H += J.transpose() * J;
    ne.g += J.transpose() * (-B.transpose() * n);
  }
  // Without baseline rho is unobservable (zero column): the damping keeps it put.
  ne.H.diagonal().array() += 1e-12;
  return ne;
}

// Gauss-Newton on x.
Eigen::Vector3d refineInverseDepth(const std::vector<Sophus::SE3d>& T_CA, const std::vector<Eigen::Vector3d>& b,
                                   Eigen::Vector3d x) {
  for (int it = 0; it < 5; ++it) {
    const NormalEquations ne = normalEquations(T_CA, b, x);
    const Eigen::Vector3d dx = ne.H.ldlt().solve(ne.g);
    x += dx;
    if (dx.norm() < 1e-9 * (1.0 + x.norm())) break;
  }
  return x;
}

}  // namespace

bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, AnchoredPoint* p,
                 double* parallax_deg) {
  std::vector<Eigen::Vector3d> d(b.size());
  for (size_t i = 0; i < b.size(); ++i) d[i] = T_GC[i].so3() * b[i];
  *parallax_deg = maxRayAngleDeg(d);
  p->T_GA = anchorFrame(T_GC.front(), b.front());
  std::vector<Sophus::SE3d> T_CA;
  for (const Sophus::SE3d& T : T_GC) T_CA.push_back(T.inverse() * p->T_GA);
  p->x = refineInverseDepth(T_CA, b, initialInverseDepth(T_GC, d, p->T_GA));
  // rho < 0: rays that meet behind the cameras. Not left to the chi-square gate,
  // whose nullspace projection removes exactly the rho direction.
  if (!p->x.allFinite() || p->x.z() < 0.0) return false;
  for (size_t i = 0; i < b.size(); ++i) {
    if ((T_GC[i].so3().inverse() * p->scaledFrom(T_GC[i].translation())).dot(b[i]) <= 0.0) return false;
  }
  return true;
}

}  // namespace msckf
