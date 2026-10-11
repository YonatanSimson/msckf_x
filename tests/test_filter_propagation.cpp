// Filter core (Stage 3): the IMU transition matches finite differences of the
// nonlinear propagation, the propagated covariance matches Monte Carlo noise,
// the clone's time-offset column matches finite differences, and clone
// marginalization keeps the other blocks of P; propagation never goes back.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/eval/nees.h"
#include "msckf/filter/propagator.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <Eigen/Cholesky>

namespace msckf::filter {
namespace {

using namespace imu_err;
using Vector15d = Eigen::Matrix<double, kDim, 1>;

const Eigen::Vector3d kTrueBg(0.01, -0.02, 0.015);
const Eigen::Vector3d kTrueBa(0.1, -0.05, 0.08);

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

// IMU only (no landmarks), noise and bias walk optional.
SimData simulate(Scenario scenario, bool noise, uint64_t seed, double duration = 3.0, double imu_rate = 0.0) {
  SimConfig c = defaultSimConfig(scenario);
  c.duration = duration;
  if (imu_rate > 0.0) c.imu_rate = imu_rate;
  c.imu_noise_enabled = noise;
  c.gyro_bias0 = kTrueBg;
  c.acc_bias0 = kTrueBa;
  c.cam_rate = 1.0;
  c.ground_density = c.wall_density = c.ceiling_density = 0.0;
  c.seed = seed;
  return Simulator(c, makeScenario(scenario, duration + 1.0, 7), camera()).run();
}

const GroundTruth& truthAt(const SimData& d, double t) {
  for (const auto& g : d.imu_truth) {
    if (g.t >= t - 1e-9) return g;
  }
  return d.imu_truth.back();
}

FilterState stateAt(const GroundTruth& g, const StateOptions& options = {}) {
  ImuInitState init;
  init.t = g.t;
  init.nav = g.state;
  init.gyro_bias = g.gyro_bias;
  init.acc_bias = g.acc_bias;
  const Eigen::Matrix<double, 7, 7> calib = 1e-6 * Eigen::Matrix<double, 7, 7>::Identity();
  return FilterState(init, forwardCameraRotation(0.2), Eigen::Vector3d(0.05, -0.03, 0.02), 0.0, options, calib);
}

// IMU error e with true = estimate (+) e.
Vector15d imuError(const FilterState& est, const FilterState& truth) {
  Vector15d e;
  e.segment<3>(kTheta) = (est.nav.R_GI.inverse() * truth.nav.R_GI).log();
  e.segment<3>(kP) = truth.nav.p_GI - est.nav.p_GI;
  e.segment<3>(kV) = truth.nav.v_GI - est.nav.v_GI;
  e.segment<3>(kBg) = truth.gyro_bias - est.gyro_bias;
  e.segment<3>(kBa) = truth.acc_bias - est.acc_bias;
  return e;
}

// Relative error of the transition against central differences of the
// nonlinear (RK4) propagation, over the given interval.
double transitionError(Scenario scenario, double t0, double T, double imu_rate = 0.0) {
  const SimData d = simulate(scenario, false, 1, 3.0, imu_rate);
  const ImuNoise noise;
  const FilterState x0 = stateAt(truthAt(d, t0), {.fej = false});
  FilterState x1 = x0;
  propagate(x1, d.imu, t0 + T, noise);
  ImuPreintegration pre(x0.gyro_bias, x0.acc_bias, noise);
  pre.integrateRange(d.imu, t0, t0 + T);
  const Matrix15d Phi = imuTransition(pre, x0.nav, x1.nav, noise).Phi;

  Matrix15d fd;
  const double eps = 1e-6;
  for (int j = 0; j < kDim; ++j) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(x0.dim());
    dx(j) = eps;
    FilterState plus = x0, minus = x0;
    plus.inject(dx);
    minus.inject(-dx);
    propagate(plus, d.imu, t0 + T, noise);
    propagate(minus, d.imu, t0 + T, noise);
    fd.col(j) = (imuError(x1, plus) - imuError(x1, minus)) / (2.0 * eps);
  }
  return (fd - Phi).norm() / Phi.norm();
}

// Over one camera interval (50 ms) the transition is exact to < 1e-6. Over
// longer intervals the preintegration's bias Jacobians (midpoint steps) and the
// RK4 nominal state part at second order in the IMU period: the mismatch must
// fall about fourfold when the IMU rate doubles.
TEST(FilterPropagation, TransitionMatchesFiniteDifferences) {
  for (Scenario s : {Scenario::kHandheld, Scenario::kFastRotation, Scenario::kCart}) {
    EXPECT_LT(transitionError(s, 1.0, 0.05), 1e-6) << toString(s);
    const double e400 = transitionError(s, 1.0, 0.5, 400.0), e800 = transitionError(s, 1.0, 0.5, 800.0);
    std::cout << toString(s) << " 0.5 s: " << e400 << " at 400 Hz, " << e800 << " at 800 Hz\n";
    EXPECT_LT(e400, 1e-5) << toString(s);
    EXPECT_LT(e800, 0.35 * e400) << toString(s);
  }
}

// Average NEES of the 15-dim IMU error after 1 s of propagation in 20 Hz
// steps, from the true state with P = 0, over Monte Carlo IMU noise.
class PropagationMonteCarlo : public ::testing::TestWithParam<Scenario> {};

TEST_P(PropagationMonteCarlo, CovarianceMatchesMonteCarlo) {
  const int runs = 200;
  const double t0 = 1.0, t1 = 2.0;
  double sum = 0.0;
  for (int r = 0; r < runs; ++r) {
    const SimData d = simulate(GetParam(), true, 100 + r);
    FilterState x = stateAt(truthAt(d, t0));
    x.P.setZero();
    for (double t = t0 + 0.05; t < t1 + 1e-9; t += 0.05) propagate(x, d.imu, t, ImuNoise{});
    FilterState truth = stateAt(truthAt(d, x.t));
    const Vector15d e = imuError(x, truth);
    const Matrix15d P = x.P.topLeftCorner<kDim, kDim>();
    sum += e.dot(P.ldlt().solve(e));
  }
  const double nees = sum / runs;
  const eval::NeesBand band = eval::neesBand95(kDim, runs);
  EXPECT_GT(nees, band.lower);
  EXPECT_LT(nees, band.upper);
  std::cout << toString(GetParam()) << " IMU NEES " << nees << " band [" << band.lower << ", " << band.upper << "]\n";
}

INSTANTIATE_TEST_SUITE_P(Scenarios, PropagationMonteCarlo,
                         ::testing::Values(Scenario::kHandheld, Scenario::kCart, Scenario::kFastRotation));

// d(clone)/d(t_d) by central differences of propagating to t_cam + t_d.
TEST(FilterPropagation, CloneTimeOffsetColumn) {
  const SimData d = simulate(Scenario::kHandheld, false, 1);
  const double t_cam = 1.3, eps = 1e-5;
  const FilterState x0 = stateAt(truthAt(d, 1.0));
  const auto cloneAt = [&](double t_d) {
    FilterState x = x0;
    x.t_d = t_d;
    const Eigen::Vector3d omega = propagate(x, d.imu, t_cam + t_d, ImuNoise{});
    x.addClone(t_cam, omega);
    return x;
  };
  const FilterState x = cloneAt(0.0);
  const FilterState plus = cloneAt(eps), minus = cloneAt(-eps);
  const Clone& c = x.clones().back();
  Eigen::Matrix<double, 6, 1> fd;
  fd.head<3>() = (c.R_GI.inverse() * plus.clones().back().R_GI).log() -
                 (c.R_GI.inverse() * minus.clones().back().R_GI).log();
  fd.tail<3>() = plus.clones().back().p_GI - minus.clones().back().p_GI;
  fd /= 2.0 * eps;

  // The analytic column, read back from the cross covariance with t_d.
  const int td = x.timeOffsetIndex();
  const Eigen::Matrix<double, 6, 1> col = x.P.block<6, 1>(x.cloneIndex(0), td) / x.P(td, td);
  EXPECT_LT((fd - col).norm(), 1e-6 * fd.norm()) << "fd " << fd.transpose() << "\ncol " << col.transpose();
}

// A target before the state time would clone the pose at the wrong time:
// refused. The same time is a no-op.
TEST(FilterPropagation, RejectsEarlierTarget) {
  const SimData d = simulate(Scenario::kHandheld, false, 1);
  FilterState x = stateAt(truthAt(d, 1.0));
  propagate(x, d.imu, 1.05, ImuNoise{});
  const Eigen::MatrixXd P = x.P;
  EXPECT_NO_THROW(propagate(x, d.imu, 1.05, ImuNoise{}));
  EXPECT_EQ(x.P, P);
  EXPECT_THROW(propagate(x, d.imu, 1.04, ImuNoise{}), std::invalid_argument);
}

TEST(FilterPropagation, MarginalizeKeepsOtherBlocks) {
  const SimData d = simulate(Scenario::kHandheld, false, 1);
  FilterState x = stateAt(truthAt(d, 1.0));
  for (int k = 1; k <= 3; ++k) {
    const Eigen::Vector3d omega = propagate(x, d.imu, 1.0 + 0.05 * k, ImuNoise{});
    x.addClone(1.0 + 0.05 * k, omega);
  }
  const Eigen::MatrixXd P = x.P;
  const int c1 = x.cloneIndex(1), n = x.dim();
  x.marginalizeClone(1);
  ASSERT_EQ(x.dim(), n - 6);
  ASSERT_EQ(x.clones().size(), 2u);
  EXPECT_DOUBLE_EQ(x.clones()[1].t_cam, 1.15);
  EXPECT_EQ(x.P.topLeftCorner(c1, c1), P.topLeftCorner(c1, c1));
  EXPECT_EQ(x.P.bottomRightCorner(6, 6), P.bottomRightCorner(6, 6));
  EXPECT_EQ(x.P.bottomLeftCorner(6, c1), P.bottomLeftCorner(6, c1));
}

}  // namespace
}  // namespace msckf::filter
