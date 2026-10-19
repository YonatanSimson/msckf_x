#include "msckf/sim/spline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msckf {

namespace {
// Cumulative basis of the uniform cubic B-spline: B~(u) = C [1 u u^2 u^3]^T.
// Row 0 is identically 1 (the base control point) and is unused.
const Eigen::Matrix4d& cumulativeBasis() {
  static const Eigen::Matrix4d C = [] {
    Eigen::Matrix4d m;
    m << 6, 0, 0, 0,
         5, 3, -3, 1,
         1, 3, 3, -2,
         0, 0, 0, 1;
    return Eigen::Matrix4d(m / 6.0);
  }();
  return C;
}
}  // namespace

void SplineTrajectory::addControlPose(const Sophus::SO3d& R_GI, const Eigen::Vector3d& p_GI) {
  rotations_.push_back(R_GI);
  positions_.push_back(p_GI);
}

TrajectoryState SplineTrajectory::evaluate(double t) const {
  const int n = numControlPoses();
  if (n < 4) throw std::logic_error("SplineTrajectory: need at least 4 control poses");
  if (t < minTime() - 1e-12 || t > maxTime() + 1e-12) {
    throw std::out_of_range("SplineTrajectory: time outside spline range");
  }

  const double s = (t - t0_) / dt_;
  const int i = std::clamp(static_cast<int>(std::floor(s)), 0, n - 4);
  const double u = s - i;

  const Eigen::Matrix4d& C = cumulativeBasis();
  const Eigen::Vector4d B = C * Eigen::Vector4d(1.0, u, u * u, u * u * u);
  const Eigen::Vector4d dB = C * Eigen::Vector4d(0.0, 1.0, 2.0 * u, 3.0 * u * u) / dt_;
  const Eigen::Vector4d ddB = C * Eigen::Vector4d(0.0, 0.0, 2.0, 6.0 * u) / (dt_ * dt_);

  TrajectoryState out;
  out.p_GI = positions_[i];
  out.v_GI.setZero();
  out.a_GI.setZero();
  out.R_GI = rotations_[i];
  out.omega_I.setZero();
  out.alpha_I.setZero();

  for (int j = 1; j <= 3; ++j) {
    const Eigen::Vector3d dp = positions_[i + j] - positions_[i + j - 1];
    out.p_GI += B[j] * dp;
    out.v_GI += dB[j] * dp;
    out.a_GI += ddB[j] * dp;

    // R <- R A_j with A_j = Exp(B_j d_j). The body rate of A_j alone is
    // dB_j d_j, so w_j = A_j^T w_{j-1} + dB_j d_j, and differentiating,
    // alpha_j = A_j^T alpha_{j-1} + (A_j^T w_{j-1}) x (dB_j d_j) + ddB_j d_j.
    const Eigen::Vector3d d = (rotations_[i + j - 1].inverse() * rotations_[i + j]).log();
    const Sophus::SO3d A = Sophus::SO3d::exp(B[j] * d);
    const Sophus::SO3d A_inv = A.inverse();
    const Eigen::Vector3d rate = dB[j] * d;
    const Eigen::Vector3d w_rot = A_inv * out.omega_I;
    out.alpha_I = A_inv * out.alpha_I + w_rot.cross(rate) + ddB[j] * d;
    out.omega_I = w_rot + rate;
    out.R_GI = out.R_GI * A;
  }
  return out;
}

}  // namespace msckf
