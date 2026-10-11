#pragma once

#include "msckf/imu/imu_types.h"

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <vector>

namespace msckf {

using Matrix9d = Eigen::Matrix<double, 9, 9>;

// On-manifold IMU preintegration (Forster, Carlone, Dellaert, Scaramuzza,
// T-RO 2017) between two times t_i and t_j, at a fixed bias linearization
// point (b_g, b_a):
//
//   dR_ij = prod_k Exp((w_k - b_g) dt_k)
//   dv_ij = sum_k dR_ik (a_k - b_a) dt_k
//   dp_ij = sum_k [dv_ik dt_k + 1/2 dR_ik (a_k - b_a) dt_k^2]
//
// all in the IMU frame at t_i, so that with gravity g (constants.h)
//
//   R_j = R_i dR_ij
//   v_j = v_i + g T + R_i dv_ij
//   p_j = p_i + v_i T + 1/2 g T^2 + R_i dp_ij,   T = t_j - t_i.
//
// Each step holds (w_k, a_k) constant over dt_k; integrateRange() feeds it
// the average of the interval's end samples. Two step schemes:
//  - kForster: dR_ik (start of the step) rotates a_k, as in the paper and in
//    GTSAM's ManifoldPreintegration, so the two compare step for step. The
//    rotation within the step is ignored, which makes it first order: 1.6 mm
//    after 1 s of handheld motion at 400 Hz, 14 mm under fast rotation.
//  - kMidpoint (default): dR_ik Exp(w_k dt_k / 2) rotates a_k. Second order;
//    the error and bias Jacobians include the half-step rotation.
//
// Errors (true = estimate (+) error), ordered like the filter state:
//   dR_true = dR Exp(dtheta), dp_true = dp + delta_p, dv_true = dv + delta_v,
// all in the frame at t_i. covariance() is their 9x9 covariance from the
// white measurement noise; the bias Jacobians give the first-order change of
// (dR, dv, dp) with the bias, used by the corrected accessors.
class ImuPreintegration {
 public:
  enum class Scheme { kForster, kMidpoint };

  ImuPreintegration(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias,
                    const ImuNoise& noise, Scheme scheme = Scheme::kMidpoint);

  // One step with measurements held constant over dt.
  void integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double dt);

  // All IMU data between t0 and t1 (t0 < t1, both inside the samples' time
  // range): each interval between consecutive samples, clipped to [t0, t1],
  // uses the average of the two samples linearly interpolated at its ends.
  void integrateRange(const std::vector<ImuSample>& imu, double t0, double t1);

  double deltaTime() const { return dt_; }
  const Sophus::SO3d& deltaR() const { return dR_; }
  const Eigen::Vector3d& deltaV() const { return dv_; }
  const Eigen::Vector3d& deltaP() const { return dp_; }
  const Matrix9d& covariance() const { return cov_; }  // [dtheta, delta_p, delta_v]

  const Eigen::Vector3d& gyroBias() const { return bg_; }
  const Eigen::Vector3d& accBias() const { return ba_; }

  const Eigen::Matrix3d& dR_dbg() const { return dR_dbg_; }
  const Eigen::Matrix3d& dV_dbg() const { return dv_dbg_; }
  const Eigen::Matrix3d& dV_dba() const { return dv_dba_; }
  const Eigen::Matrix3d& dP_dbg() const { return dp_dbg_; }
  const Eigen::Matrix3d& dP_dba() const { return dp_dba_; }

  // Deltas corrected to first order for a different bias (Forster eq. 44).
  Sophus::SO3d deltaR(const Eigen::Vector3d& gyro_bias) const;
  Eigen::Vector3d deltaV(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias) const;
  Eigen::Vector3d deltaP(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias) const;

  // State at t_j from the state at t_i, with the deltas corrected to the given
  // bias.
  NavState predict(const NavState& state_i, const Eigen::Vector3d& gyro_bias,
                   const Eigen::Vector3d& acc_bias) const;

 private:
  Eigen::Vector3d bg_, ba_;
  ImuNoise noise_;
  Scheme scheme_;

  double dt_ = 0.0;
  Sophus::SO3d dR_;
  Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp_ = Eigen::Vector3d::Zero();
  Matrix9d cov_ = Matrix9d::Zero();

  Eigen::Matrix3d dR_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba_ = Eigen::Matrix3d::Zero();
};

// Right Jacobian of SO(3): Exp(phi + d) ~ Exp(phi) Exp(Jr(phi) d).
Eigen::Matrix3d so3RightJacobian(const Eigen::Vector3d& phi);

}  // namespace msckf
