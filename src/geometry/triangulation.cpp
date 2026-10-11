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

bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, Eigen::Vector3d* X,
                 double* parallax_deg) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> d(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    d[i] = T_GC[i].so3() * b[i];
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d[i] * d[i].transpose();
    A += P;
    rhs += P * T_GC[i].translation();
  }
  *parallax_deg = maxRayAngleDeg(d);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.eigenvalues()(0) < 1e-9 * es.eigenvalues()(2)) return false;
  *X = A.ldlt().solve(rhs);
  for (int it = 0; it < 5; ++it) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < b.size(); ++i) {
      const Eigen::Vector3d pc = T_GC[i].inverse() * *X;
      const double r = pc.norm();
      const Eigen::Vector3d n = pc / r;
      const Matrix32d B = tangentBasis(b[i]);
      const Eigen::Matrix<double, 2, 3> J =
          B.transpose() * (Eigen::Matrix3d::Identity() - n * n.transpose()) / r * T_GC[i].so3().inverse().matrix();
      const Eigen::Vector2d res = -B.transpose() * n;
      H += J.transpose() * J;
      g += J.transpose() * res;
    }
    const Eigen::Vector3d dX = H.ldlt().solve(g);
    *X += dX;
    if (dX.norm() < 1e-9 * (1.0 + X->norm())) break;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    if ((T_GC[i].inverse() * *X).dot(b[i]) <= 0.0) return false;
  }
  return true;
}

}  // namespace msckf
