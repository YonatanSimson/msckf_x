// IMU preintegration: predicts the simulator's exact state, bias Jacobians
// match finite differences, and the covariance matches Monte Carlo noise.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/eval/nees.h"
#include "msckf/imu/preintegration.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <Eigen/Cholesky>

#include <random>

namespace msckf {
namespace {

const Eigen::Vector3d kBg(0.01, -0.02, 0.015);
const Eigen::Vector3d kBa(0.1, -0.05, 0.08);

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05,
      0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

// Noise-free IMU with constant biases, plus the exact states at the samples.
SimData simulate(Scenario scenario, double rate, double duration = 6.0) {
  SimConfig c = defaultSimConfig(scenario);
  c.duration = duration;
  c.imu_rate = rate;
  c.imu_noise_enabled = false;
  c.gyro_bias0 = kBg;
  c.acc_bias0 = kBa;
  c.cam_rate = 1.0;
  c.ground_density = c.wall_density = c.ceiling_density = 0.0;
  return Simulator(c, makeScenario(scenario, duration + 1.0, 7), camera()).run();
}

struct PredictionError {
  double rot = 0.0, pos = 0.0, vel = 0.0;
};

// Max prediction error over windows [t, t + window] across the run.
PredictionError maxPredictionError(const SimData& d, double window,
                                   ImuPreintegration::Scheme scheme) {
  PredictionError e;
  const double t_max = d.imu.back().t - window;
  for (double t0 = 0.5; t0 < t_max; t0 += 0.25) {
    const auto at = [&](double t) -> const GroundTruth& {
      for (const auto& g : d.imu_truth) {
        if (g.t >= t - 1e-9) return g;
      }
      return d.imu_truth.back();
    };
    const GroundTruth& gi = at(t0);
    const GroundTruth& gj = at(t0 + window);
    ImuPreintegration pre(kBg, kBa, ImuNoise{}, scheme);
    pre.integrateRange(d.imu, gi.t, gj.t);
    const NavState pj = pre.predict(gi.state, kBg, kBa);
    e.rot = std::max(e.rot, (pj.R_GI.inverse() * gj.state.R_GI).log().norm());
    e.pos = std::max(e.pos, (pj.p_GI - gj.state.p_GI).norm());
    e.vel = std::max(e.vel, (pj.v_GI - gj.state.v_GI).norm());
  }
  return e;
}

TEST(Preintegration, PredictsExactStateAndConverges) {
  using Scheme = ImuPreintegration::Scheme;
  for (Scenario s : {Scenario::kHandheld, Scenario::kCart, Scenario::kFastRotation}) {
    const SimData d400 = simulate(s, 400.0), d800 = simulate(s, 800.0);
    const PredictionError m400 = maxPredictionError(d400, 1.0, Scheme::kMidpoint);
    const PredictionError m800 = maxPredictionError(d800, 1.0, Scheme::kMidpoint);
    const PredictionError f400 = maxPredictionError(d400, 1.0, Scheme::kForster);
    const PredictionError f800 = maxPredictionError(d800, 1.0, Scheme::kForster);
    std::cout << "[ info ] " << toString(s) << ", 1 s windows at 400 Hz: midpoint rot " << m400.rot
              << " rad, pos " << m400.pos << " m, vel " << m400.vel << " m/s (800 Hz ratio "
              << m400.pos / m800.pos << "); Forster pos " << f400.pos << " m (ratio "
              << f400.pos / f800.pos << ")\n";
    // Midpoint: second order, halving dt cuts the error about four times.
    EXPECT_LT(m400.pos, 1e-4) << toString(s);
    EXPECT_LT(m400.rot, 1e-4) << toString(s);
    EXPECT_GT(m400.pos / m800.pos, 3.5) << toString(s);
    EXPECT_GT(m400.rot / m800.rot, 3.5) << toString(s);
    // Forster/GTSAM step: first order in position and velocity.
    EXPECT_NEAR(f400.pos / f800.pos, 2.0, 0.2) << toString(s);
  }
}

TEST(Preintegration, BiasJacobiansMatchFiniteDifferences) {
  const SimData d = simulate(Scenario::kFastRotation, 400.0, 3.0);
  const double t0 = 0.5, t1 = 1.5;
  for (auto scheme : {ImuPreintegration::Scheme::kMidpoint, ImuPreintegration::Scheme::kForster}) {
  ImuPreintegration ref(kBg, kBa, ImuNoise{}, scheme);
  ref.integrateRange(d.imu, t0, t1);

  constexpr double h = 1e-6;
  double worst = 0.0;
  for (int k = 0; k < 6; ++k) {
    Eigen::Vector3d bg = kBg, ba = kBa;
    (k < 3 ? bg[k] : ba[k - 3]) += h;
    ImuPreintegration p(bg, ba, ImuNoise{}, scheme);
    p.integrateRange(d.imu, t0, t1);
    const Eigen::Vector3d dR_fd = (ref.deltaR().inverse() * p.deltaR()).log() / h;
    const Eigen::Vector3d dV_fd = (p.deltaV() - ref.deltaV()) / h;
    const Eigen::Vector3d dP_fd = (p.deltaP() - ref.deltaP()) / h;
    const int c = k % 3;
    const Eigen::Vector3d dR_an = k < 3 ? Eigen::Vector3d(ref.dR_dbg().col(c)) : Eigen::Vector3d::Zero();
    const Eigen::Vector3d dV_an = k < 3 ? ref.dV_dbg().col(c) : ref.dV_dba().col(c);
    const Eigen::Vector3d dP_an = k < 3 ? ref.dP_dbg().col(c) : ref.dP_dba().col(c);
    worst = std::max({worst, (dR_fd - dR_an).norm() / std::max(dR_an.norm(), 1e-3),
                      (dV_fd - dV_an).norm() / dV_an.norm(), (dP_fd - dP_an).norm() / dP_an.norm()});
  }
  std::cout << "[ info ] bias Jacobians vs finite differences: max rel. error " << worst << "\n";
  EXPECT_LT(worst, 1e-4);

  // First-order bias correction vs re-integration for a 1e-3 bias change:
  // the residual must be second order (~1e-6 relative).
  const Eigen::Vector3d dbg(1e-3, -1e-3, 5e-4), dba(1e-3, 2e-3, -1e-3);
  ImuPreintegration p(kBg + dbg, kBa + dba, ImuNoise{}, scheme);
  p.integrateRange(d.imu, t0, t1);
  EXPECT_LT((ref.deltaR(kBg + dbg).inverse() * p.deltaR()).log().norm(), 1e-5);
  EXPECT_LT((ref.deltaV(kBg + dbg, kBa + dba) - p.deltaV()).norm(), 1e-5);
  EXPECT_LT((ref.deltaP(kBg + dbg, kBa + dba) - p.deltaP()).norm(), 1e-5);
  }
}

class PreintegrationCovariance : public ::testing::TestWithParam<ImuPreintegration::Scheme> {};

TEST_P(PreintegrationCovariance, MatchesMonteCarlo) {
  constexpr int kRuns = 500;
  const SimData d = simulate(Scenario::kHandheld, 200.0, 3.0);
  // Samples in [0.5, 1.5] held over their interval, exactly as integrate() models.
  std::vector<ImuSample> s;
  for (const auto& x : d.imu) {
    if (x.t >= 0.5 - 1e-9 && x.t <= 1.5 + 1e-9) s.push_back(x);
  }
  ImuNoise noise;
  noise.gyro_noise = 1e-3;
  noise.acc_noise = 1e-2;
  auto run = [&](std::mt19937_64* rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    ImuPreintegration p(kBg, kBa, noise, GetParam());
    for (size_t k = 0; k + 1 < s.size(); ++k) {
      const double dt = s[k + 1].t - s[k].t;
      Eigen::Vector3d g = s[k].gyro, a = s[k].acc;
      if (rng) {
        g += noise.gyro_noise / std::sqrt(dt) * Eigen::Vector3d(n(*rng), n(*rng), n(*rng));
        a += noise.acc_noise / std::sqrt(dt) * Eigen::Vector3d(n(*rng), n(*rng), n(*rng));
      }
      p.integrate(g, a, dt);
    }
    return p;
  };
  const ImuPreintegration nominal = run(nullptr);
  const auto llt = nominal.covariance().ldlt();
  std::mt19937_64 rng(3);
  double nees = 0.0;
  for (int r = 0; r < kRuns; ++r) {
    const ImuPreintegration p = run(&rng);
    Eigen::Matrix<double, 9, 1> e;  // true - estimate, [dtheta, delta_p, delta_v]
    e << (nominal.deltaR().inverse() * p.deltaR()).log(), p.deltaP() - nominal.deltaP(),
        p.deltaV() - nominal.deltaV();
    nees += e.dot(llt.solve(e));
  }
  nees /= kRuns;
  const auto band = eval::neesBand95(9, kRuns);
  std::cout << "[ info ] preintegration NEES (9 dof) " << nees << " in [" << band.lower << ", "
            << band.upper << "]\n";
  EXPECT_GT(nees, band.lower);
  EXPECT_LT(nees, band.upper);
}

INSTANTIATE_TEST_SUITE_P(Schemes, PreintegrationCovariance,
                         ::testing::Values(ImuPreintegration::Scheme::kMidpoint,
                                           ImuPreintegration::Scheme::kForster),
                         [](const auto& info) {
                           return info.param == ImuPreintegration::Scheme::kMidpoint ? "Midpoint"
                                                                                       : "Forster";
                         });

}  // namespace
}  // namespace msckf
