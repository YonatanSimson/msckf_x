#include "msckf/filter/propagator.h"

#include "msckf/core/constants.h"
#include "msckf/imu/propagation.h"

#include <algorithm>
#include <stdexcept>

namespace msckf::filter {

namespace {

// IMU reading at time t, linearly interpolated between samples a and b.
ImuReading readingAt(const ImuSample& a, const ImuSample& b, double t, const Eigen::Vector3d& bg,
                     const Eigen::Vector3d& ba) {
  const double alpha = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0.0;
  return interpolate({a.gyro - bg, a.acc - ba}, {b.gyro - bg, b.acc - ba}, alpha);
}

}  // namespace

ImuTransition imuTransition(const ImuPreintegration& pre, const NavState& lin_k, const NavState& x_k1,
                            const ImuNoise& noise) {
  using namespace imu_err;
  const double T = pre.deltaTime();
  const Eigen::Matrix3d R_k = lin_k.R_GI.matrix();
  const Eigen::Vector3d Rdv = x_k1.v_GI - lin_k.v_GI - kGravity * T;
  const Eigen::Vector3d Rdp = x_k1.p_GI - lin_k.p_GI - lin_k.v_GI * T - 0.5 * kGravity * T * T;

  ImuTransition tr;
  Matrix15d& F = tr.Phi;
  F.block<3, 3>(kTheta, kTheta) = x_k1.R_GI.matrix().transpose() * R_k;
  F.block<3, 3>(kTheta, kBg) = pre.dR_dbg();
  F.block<3, 3>(kV, kTheta) = -Sophus::SO3d::hat(Rdv) * R_k;
  F.block<3, 3>(kV, kBg) = R_k * pre.dV_dbg();
  F.block<3, 3>(kV, kBa) = R_k * pre.dV_dba();
  F.block<3, 3>(kP, kTheta) = -Sophus::SO3d::hat(Rdp) * R_k;
  F.block<3, 3>(kP, kV) = T * Eigen::Matrix3d::Identity();
  F.block<3, 3>(kP, kBg) = R_k * pre.dP_dbg();
  F.block<3, 3>(kP, kBa) = R_k * pre.dP_dba();

  // Preintegration covariance is [dtheta, dp, dv] in the frame at t_k, the
  // same order as the state's first nine entries.
  Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Identity();
  G.block<3, 3>(3, 3) = R_k;
  G.block<3, 3>(6, 6) = R_k;
  tr.Q.topLeftCorner<9, 9>() = G * pre.covariance() * G.transpose();
  tr.Q.block<3, 3>(kBg, kBg).diagonal().setConstant(noise.gyro_bias_walk * noise.gyro_bias_walk * T);
  tr.Q.block<3, 3>(kBa, kBa).diagonal().setConstant(noise.acc_bias_walk * noise.acc_bias_walk * T);
  return tr;
}

Eigen::Vector3d propagate(FilterState& s, const std::vector<ImuSample>& imu, double t, const ImuNoise& noise) {
  using namespace imu_err;
  if (imu.size() < 2 || s.t < imu.front().t || t > imu.back().t) {
    throw std::out_of_range("filter::propagate: IMU samples do not cover the interval");
  }
  const auto later = [](double v, const ImuSample& x) { return v < x.t; };
  // Index of the sample interval [i, i + 1] containing time v.
  const auto interval = [&](double v) {
    const auto it = std::upper_bound(imu.begin(), imu.end(), v, later);
    return std::clamp<long>(static_cast<long>(it - imu.begin()) - 1, 0, static_cast<long>(imu.size()) - 2);
  };
  const auto omegaAt = [&](double v) {
    const long i = interval(v);
    return readingAt(imu[i], imu[i + 1], v, s.gyro_bias, s.acc_bias).omega;
  };
  if (t < s.t) {
    // The clone at t would be the pose at s.t: refuse instead of mislabelling it.
    throw std::invalid_argument("filter::propagate: target time precedes the state time");
  }
  if (t == s.t) {
    s.Phi_last.setIdentity();
    return omegaAt(s.t);
  }

  ImuPreintegration pre(s.gyro_bias, s.acc_bias, noise);
  pre.integrateRange(imu, s.t, t);

  NavState x = s.nav;
  for (long i = interval(s.t); i + 1 < static_cast<long>(imu.size()) && imu[i].t < t; ++i) {
    const double lo = std::max(imu[i].t, s.t), hi = std::min(imu[i + 1].t, t);
    if (hi <= lo) continue;
    const auto r = [&](double v) { return readingAt(imu[i], imu[i + 1], v, s.gyro_bias, s.acc_bias); };
    x = rk4Step(x, r(lo), r(0.5 * (lo + hi)), r(hi), hi - lo);
  }

  const ImuTransition tr = imuTransition(pre, s.options().fej ? s.nav_fej : s.nav, x, noise);
  const int rest = s.dim() - kDim;
  s.P.topLeftCorner<kDim, kDim>() =
      tr.Phi * s.P.topLeftCorner<kDim, kDim>() * tr.Phi.transpose() + tr.Q;
  if (rest > 0) {
    s.P.topRightCorner(kDim, rest) = (tr.Phi * s.P.topRightCorner(kDim, rest)).eval();
    s.P.bottomLeftCorner(rest, kDim) = s.P.topRightCorner(kDim, rest).transpose();
  }
  s.Phi_last = tr.Phi;
  s.nav = x;
  s.nav_fej = x;
  s.t = t;
  return omegaAt(t);
}

}  // namespace msckf::filter
