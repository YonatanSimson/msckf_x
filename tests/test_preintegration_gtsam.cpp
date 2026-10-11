// Cross-check of ImuPreintegration (kForster scheme) against GTSAM's
// ManifoldPreintegration (BSD), which implements the same Forster et al.
// discretization: deltas, bias Jacobians, covariance and prediction must agree
// step for step to numerical precision. GTSAM is a test-only reference
// (scripts/deps/build_gtsam.sh); this test is built only when it is found.
//
// Convention bridge: GTSAM's covariance is in NavState local coordinates,
// where position and velocity errors are in the body frame of dR_ij
// (p = p_hat + dR_ij delta_p_body); ours are in the frame at t_i
// (p = p_hat + delta_p). Rotation errors are both right-multiplied.

#include <gtest/gtest.h>

#include "msckf/imu/preintegration.h"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ManifoldPreintegration.h>

#include <random>

namespace msckf {
namespace {

using GtsamPim = gtsam::PreintegratedImuMeasurementsT<gtsam::ManifoldPreintegration>;

double relErr(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).norm() / std::max(b.norm(), 1e-300);
}

TEST(PreintegrationGtsam, MatchesManifoldPreintegrationStepForStep) {
  ImuNoise noise;
  noise.gyro_noise = 1.7e-4;
  noise.acc_noise = 2.0e-3;
  const Eigen::Vector3d bg(0.01, -0.02, 0.005), ba(0.1, 0.05, -0.08);

  auto params = gtsam::PreintegrationParams::MakeSharedU(9.81);
  params->setGyroscopeCovariance(noise.gyro_noise * noise.gyro_noise * Eigen::Matrix3d::Identity());
  params->setAccelerometerCovariance(noise.acc_noise * noise.acc_noise * Eigen::Matrix3d::Identity());
  params->setIntegrationCovariance(Eigen::Matrix3d::Zero());
  GtsamPim gt(params, gtsam::imuBias::ConstantBias(ba, bg));
  ImuPreintegration ours(bg, ba, noise, ImuPreintegration::Scheme::kForster);

  // 1 s of aggressive random motion with jittered sample intervals.
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Eigen::Vector3d w(0.5, -1.0, 2.0), a(1.0, 2.0, 9.0);
  double worst_delta = 0.0, worst_jac = 0.0, worst_cov = 0.0;
  for (int k = 0; k < 400; ++k) {
    const double dt = 0.0025 * (1.0 + 0.2 * u(rng));
    w += 0.2 * Eigen::Vector3d(u(rng), u(rng), u(rng));
    a += 0.5 * Eigen::Vector3d(u(rng), u(rng), u(rng));
    gt.integrateMeasurement(a, w, dt);
    ours.integrate(w, a, dt);

    const Eigen::Matrix3d R = gt.deltaRij().matrix();
    worst_delta = std::max({worst_delta,
                            (ours.deltaR().inverse() * Sophus::SO3d(Sophus::makeRotationMatrix(R)))
                                .log()
                                .norm(),
                            relErr(ours.deltaV(), gt.deltaVij()), relErr(ours.deltaP(), gt.deltaPij())});
    worst_jac = std::max({worst_jac, relErr(ours.dR_dbg(), gt.delRdelBiasOmega()),
                          relErr(ours.dV_dbg(), gt.delVdelBiasOmega()),
                          relErr(ours.dV_dba(), gt.delVdelBiasAcc()),
                          relErr(ours.dP_dbg(), gt.delPdelBiasOmega()),
                          relErr(ours.dP_dba(), gt.delPdelBiasAcc())});
    // GTSAM order [R, p_body, v_body] -> ours [theta, p_i, v_i].
    Matrix9d T = Matrix9d::Identity();
    T.block<3, 3>(3, 3) = R;
    T.block<3, 3>(6, 6) = R;
    const Matrix9d cov_gt = T * gt.preintMeasCov() * T.transpose();
    worst_cov = std::max(worst_cov, relErr(ours.covariance(), cov_gt));
  }
  std::cout << "[ info ] vs GTSAM over 400 steps: deltas " << worst_delta << ", bias Jacobians "
            << worst_jac << ", covariance " << worst_cov << " (max relative)\n";
  EXPECT_LT(worst_delta, 1e-12);
  EXPECT_LT(worst_jac, 1e-12);
  EXPECT_LT(worst_cov, 1e-12);

  // Prediction with a different bias (first-order correction on both sides).
  const Eigen::Vector3d bg2 = bg + Eigen::Vector3d(1e-3, -2e-3, 5e-4);
  const Eigen::Vector3d ba2 = ba + Eigen::Vector3d(-0.01, 0.02, 0.005);
  NavState s;
  s.R_GI = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3));
  s.p_GI = Eigen::Vector3d(1.0, 2.0, 3.0);
  s.v_GI = Eigen::Vector3d(0.5, -0.3, 0.2);
  const NavState p_ours = ours.predict(s, bg2, ba2);
  const gtsam::NavState p_gt = gt.predict(
      gtsam::NavState(gtsam::Rot3(s.R_GI.matrix()), s.p_GI, s.v_GI),
      gtsam::imuBias::ConstantBias(ba2, bg2));
  const double rot = (p_ours.R_GI.inverse() *
                      Sophus::SO3d(Sophus::makeRotationMatrix(p_gt.attitude().matrix())))
                         .log()
                         .norm();
  std::cout << "[ info ] predict with corrected bias vs GTSAM: rot " << rot << " rad, pos "
            << (p_ours.p_GI - p_gt.position()).norm() << " m, vel "
            << (p_ours.v_GI - p_gt.velocity()).norm() << " m/s\n";
  EXPECT_LT(rot, 1e-9);
  EXPECT_LT((p_ours.p_GI - p_gt.position()).norm(), 1e-9);
  EXPECT_LT((p_ours.v_GI - p_gt.velocity()).norm(), 1e-9);
}

}  // namespace
}  // namespace msckf
