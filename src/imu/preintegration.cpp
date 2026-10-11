#include "msckf/imu/preintegration.h"

#include <algorithm>

#include "msckf/core/constants.h"

#include <cmath>
#include <stdexcept>

namespace msckf {

Eigen::Matrix3d so3RightJacobian(const Eigen::Vector3d& phi) {
  const double theta2 = phi.squaredNorm();
  const Eigen::Matrix3d W = Sophus::SO3d::hat(phi);
  if (theta2 < 1e-10) return Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 6.0) * W * W;
  const double theta = std::sqrt(theta2);
  return Eigen::Matrix3d::Identity() - (1.0 - std::cos(theta)) / theta2 * W +
         (theta - std::sin(theta)) / (theta2 * theta) * W * W;
}

ImuPreintegration::ImuPreintegration(const Eigen::Vector3d& gyro_bias,
                                     const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                     Scheme scheme)
    : bg_(gyro_bias), ba_(acc_bias), noise_(noise), scheme_(scheme) {}

void ImuPreintegration::integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc,
                                  double dt) {
  if (!(dt > 0.0)) throw std::invalid_argument("ImuPreintegration: dt must be positive");
  const Eigen::Vector3d w = gyro - bg_;
  const Eigen::Vector3d a = acc - ba_;
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = dR_.matrix();  // dR_ik, before this step
  const Sophus::SO3d dR_step = Sophus::SO3d::exp(w * dt);
  const Eigen::Matrix3d dR_step_T = dR_step.matrix().transpose();
  const Eigen::Matrix3d Jr = so3RightJacobian(w * dt);
  const double dt2 = 0.5 * dt * dt;

  // Rotation applied to a over the step: Ra = dR_ik E with E = I (kForster)
  // or Exp(w dt / 2) (kMidpoint). Its sensitivities, with E = Exp(c w dt):
  //   to dtheta (right error on dR_ik):  d(Ra a) = -Ra [a]x E^T dtheta
  //   to w (bias and gyro noise):        d(Ra a) = -Ra [a]x Jr(c w dt) c dt dw
  const bool mid = scheme_ == Scheme::kMidpoint;
  const double c = mid ? 0.5 : 0.0;
  const Eigen::Matrix3d E = mid ? Sophus::SO3d::exp(c * w * dt).matrix() : I3;
  const Eigen::Matrix3d Jr_c = mid ? so3RightJacobian(c * w * dt) : I3;
  const Eigen::Matrix3d Ra = R * E;
  const Eigen::Matrix3d Ra_a = Ra * Sophus::SO3d::hat(a);
  const Eigen::Matrix3d dRaa_dtheta = -Ra_a * E.transpose();
  const Eigen::Matrix3d dRaa_dw = -Ra_a * Jr_c * (c * dt);

  // Error propagation, order [dtheta, delta_p, delta_v]; w and a are the
  // bias-corrected measurements, the noise enters through them.
  //   dtheta' = dR_step^T dtheta + Jr dt dw
  //   delta_v' = delta_v + dt d(Ra a)
  //   delta_p' = delta_p + dt delta_v + dt^2 / 2 d(Ra a)
  Matrix9d A = Matrix9d::Identity();
  A.block<3, 3>(0, 0) = dR_step_T;
  A.block<3, 3>(3, 0) = dRaa_dtheta * dt2;
  A.block<3, 3>(3, 6) = I3 * dt;
  A.block<3, 3>(6, 0) = dRaa_dtheta * dt;
  Eigen::Matrix<double, 9, 3> Gw, Ga;
  Gw << Jr * dt, dRaa_dw * dt2, dRaa_dw * dt;
  Ga << Eigen::Matrix3d::Zero(), Ra * dt2, Ra * dt;
  // Discrete white noise: sigma^2 / dt for a sample held over dt.
  const double qg = noise_.gyro_noise * noise_.gyro_noise / dt;
  const double qa = noise_.acc_noise * noise_.acc_noise / dt;
  cov_ = A * cov_ * A.transpose() + qg * Gw * Gw.transpose() + qa * Ga * Ga.transpose();

  // Bias Jacobians (Forster appendix B, generalized to the step rotation Ra):
  // a bias change db_g changes dR_ik by dR_dbg db_g and w by -db_g, so
  //   d(Ra a)/db_g = dRaa_dtheta dR_dbg - dRaa_dw,  d(Ra a)/db_a = -Ra.
  // Position before velocity before rotation: each uses the previous values.
  const Eigen::Matrix3d dRaa_dbg = dRaa_dtheta * dR_dbg_ - dRaa_dw;
  dp_dba_ += dv_dba_ * dt - Ra * dt2;
  dp_dbg_ += dv_dbg_ * dt + dRaa_dbg * dt2;
  dv_dba_ -= Ra * dt;
  dv_dbg_ += dRaa_dbg * dt;
  dR_dbg_ = dR_step_T * dR_dbg_ - Jr * dt;

  // Deltas, same order.
  dp_ += dv_ * dt + Ra * a * dt2;
  dv_ += Ra * a * dt;
  dR_ = dR_ * dR_step;
  dt_ += dt;
}

void ImuPreintegration::integrateRange(const std::vector<ImuSample>& imu, double t0, double t1) {
  if (!(t1 > t0)) throw std::invalid_argument("integrateRange: t1 must be after t0");
  if (imu.size() < 2 || imu.front().t > t0 || imu.back().t < t1) {
    throw std::invalid_argument("integrateRange: IMU samples do not cover [t0, t1]");
  }
  auto lerp = [](const ImuSample& s0, const ImuSample& s1, double t, Eigen::Vector3d& g,
                 Eigen::Vector3d& a) {
    const double u = (t - s0.t) / (s1.t - s0.t);
    g = s0.gyro + u * (s1.gyro - s0.gyro);
    a = s0.acc + u * (s1.acc - s0.acc);
  };
  // First interval that can overlap [t0, t1]: the last sample at or before t0.
  auto it = std::upper_bound(imu.begin(), imu.end(), t0, [](double t, const ImuSample& x) { return t < x.t; });
  const size_t first = it == imu.begin() ? 0 : static_cast<size_t>(it - imu.begin()) - 1;
  for (size_t k = first; k + 1 < imu.size() && imu[k].t < t1; ++k) {
    const ImuSample& s0 = imu[k];
    const ImuSample& s1 = imu[k + 1];
    const double lo = std::max(s0.t, t0), hi = std::min(s1.t, t1);
    if (hi <= lo) continue;
    Eigen::Vector3d g_lo, a_lo, g_hi, a_hi;
    lerp(s0, s1, lo, g_lo, a_lo);
    lerp(s0, s1, hi, g_hi, a_hi);
    integrate(0.5 * (g_lo + g_hi), 0.5 * (a_lo + a_hi), hi - lo);
  }
}

Sophus::SO3d ImuPreintegration::deltaR(const Eigen::Vector3d& gyro_bias) const {
  return dR_ * Sophus::SO3d::exp(dR_dbg_ * (gyro_bias - bg_));
}

Eigen::Vector3d ImuPreintegration::deltaV(const Eigen::Vector3d& gyro_bias,
                                          const Eigen::Vector3d& acc_bias) const {
  return dv_ + dv_dbg_ * (gyro_bias - bg_) + dv_dba_ * (acc_bias - ba_);
}

Eigen::Vector3d ImuPreintegration::deltaP(const Eigen::Vector3d& gyro_bias,
                                          const Eigen::Vector3d& acc_bias) const {
  return dp_ + dp_dbg_ * (gyro_bias - bg_) + dp_dba_ * (acc_bias - ba_);
}

NavState ImuPreintegration::predict(const NavState& s, const Eigen::Vector3d& gyro_bias,
                                    const Eigen::Vector3d& acc_bias) const {
  NavState out;
  const double T = dt_;
  out.R_GI = s.R_GI * deltaR(gyro_bias);
  out.v_GI = s.v_GI + kGravity * T + s.R_GI * deltaV(gyro_bias, acc_bias);
  out.p_GI = s.p_GI + s.v_GI * T + 0.5 * kGravity * T * T + s.R_GI * deltaP(gyro_bias, acc_bias);
  return out;
}

}  // namespace msckf
