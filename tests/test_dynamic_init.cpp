// Stage 1b retry pipeline and hand-offs: the linear (NR) hand-off's
// covariance is consistent when the biases are drawn from their prior; the
// pipeline refuses windows the IMU does not cover or that straddle an IMU gap
// instead of throwing or interpolating; the linear solve refuses fewer than
// six equations.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/core/constants.h"
#include "msckf/eval/nees.h"
#include "msckf/init/dynamic_init.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace msckf {
namespace {

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

// Frames from t0 to t1, bearings from the noisy pixels.
std::vector<InitFrame> framesOf(const Simulator& sim, const SimData& data, double t0, double t1) {
  std::vector<InitFrame> out;
  for (const CameraFrame& f : data.frames) {
    const double t = f.t_cam + sim.config().time_offset;
    if (t < t0 - 1e-9 || t > t1 + 1e-9) continue;
    InitFrame fr;
    fr.t = t;
    for (const Observation& o : f.observations) {
      Eigen::Vector3d b;
      if (sim.camera().unproject(o.uv, b)) fr.bearings.emplace_back(o.landmark_id, b);
    }
    out.push_back(std::move(fr));
  }
  return out;
}

Sophus::SE3d T_IC(const SimConfig& c) { return Sophus::SE3d(c.R_IC, c.p_IC); }

// NEES of tilt (2) and body-frame velocity (3), both independent of the yaw
// and position gauge. Errors with truth = estimate (+) error:
//   g_b true = g_b + [g_b]x dtheta,  v_b true = v_b + [v_b]x dtheta + R^T dv.
double tiltVelocityNees(const ImuInitState& st, const NavState& truth) {
  using namespace imu_err;
  const Eigen::Matrix3d Rt = st.nav.R_GI.inverse().matrix();
  const Eigen::Vector3d g_b = Rt * kGravity, v_b = Rt * st.nav.v_GI;
  const Eigen::Vector3d u = g_b.normalized();
  const Eigen::Vector3d a = std::abs(u.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  Eigen::Matrix<double, 3, 2> B;
  B.col(0) = u.cross(a).normalized();
  B.col(1) = u.cross(B.col(0));
  Eigen::Matrix<double, 5, 15> J = Eigen::Matrix<double, 5, 15>::Zero();
  J.block<2, 3>(0, kTheta) = B.transpose() * Sophus::SO3d::hat(g_b);
  J.block<3, 3>(2, kTheta) = Sophus::SO3d::hat(v_b);
  J.block<3, 3>(2, kV) = Rt;
  const Eigen::Matrix3d Rt_true = truth.R_GI.inverse().matrix();
  Eigen::Matrix<double, 5, 1> e;
  e.head<2>() = B.transpose() * (Rt_true * kGravity - g_b);
  e.tail<3>() = Rt_true * truth.v_GI - v_b;
  const Eigen::Matrix<double, 5, 5> S = J * st.P * J.transpose();
  return e.dot(S.ldlt().solve(e));
}

// Monte Carlo over windows of handheld motion, true biases drawn from the
// prior and the prior mean (zero) given to the initializer: the hand-off's
// tilt and velocity NEES is inside its 95 % band.
TEST(LinearHandoff, ConsistentWithBiasesFromThePrior) {
  const DynamicInitConfig cfg;
  std::vector<double> nees;
  for (uint64_t seed = 1; seed <= 10; ++seed) {
    SimConfig c = defaultSimConfig(Scenario::kHandheld);
    c.duration = 21.0;
    c.seed = seed;
    std::mt19937_64 rng(200 + seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    c.gyro_bias0 = cfg.refinement.gyro_bias_sigma * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    c.acc_bias0 = cfg.refinement.acc_bias_sigma * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    const Simulator sim(c, makeScenario(Scenario::kHandheld, 22.0, seed), camera());
    const SimData data = sim.run();
    for (double t0 = 1.0; t0 + 1.0 < c.duration; t0 += 2.0) {
      std::vector<InitFrame> kf;
      const std::vector<InitFrame> all = framesOf(sim, data, t0, t0 + 1.0);
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
      for (int k : selectKeyframes(data.imu, all, T_IC(c), zero)) kf.push_back(all[k]);
      const FeaturelessInitResult lin = featurelessInit(data.imu, kf, T_IC(c), zero, zero, c.imu_noise, cfg.linear);
      if (!lin.ok) continue;
      const auto st = linearHandoff(data.imu, kf, T_IC(c), lin, zero, zero, c.imu_noise, cfg);
      if (!st) continue;
      nees.push_back(tiltVelocityNees(*st, sim.trueState(st->t)));
    }
  }
  ASSERT_GE(nees.size(), 80u);
  double mean = 0.0;
  for (double v : nees) mean += v / static_cast<double>(nees.size());
  const eval::NeesBand band = eval::neesBand95(5, static_cast<int>(nees.size()));
  std::cout << "[ info ] linear hand-off tilt + velocity NEES " << mean << " over " << nees.size() << " windows, band ["
            << band.lower << ", " << band.upper << "]\n";
  EXPECT_LE(mean, band.upper);
  EXPECT_GE(mean, band.lower);
}

struct Data {
  SimConfig c;
  SimData data;
  std::vector<InitFrame> frames;
};

Data handheld(double duration = 4.0) {
  Data d{defaultSimConfig(Scenario::kHandheld), {}, {}};
  d.c.duration = duration;
  const Simulator sim(d.c, makeScenario(Scenario::kHandheld, duration + 1.0, 1), camera());
  d.data = sim.run();
  d.frames = framesOf(sim, d.data, 0.0, duration);
  return d;
}

DynamicInitResult run(const Data& d, const std::vector<ImuSample>& imu, double t_start,
                      const DynamicInitConfig& cfg = {}) {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  return initializeDynamic(imu, d.frames, t_start, T_IC(d.c), zero, zero, d.c.imu_noise, cfg);
}

// IMU that starts after the window's first frame: refused, not thrown.
TEST(DynamicInit, RefusesWindowBeforeTheImu) {
  const Data d = handheld();
  std::vector<ImuSample> imu;
  for (const ImuSample& s : d.data.imu) {
    if (s.t >= 1.2) imu.push_back(s);
  }
  DynamicInitResult r;
  ASSERT_NO_THROW(r = run(d, imu, 1.0));
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.reason, "imu");
}

// A gap that straddles the window's first frame is found.
TEST(DynamicInit, RefusesGapAtWindowStart) {
  const Data d = handheld();
  const double t_first = d.frames[20].t;
  std::vector<ImuSample> imu;
  for (const ImuSample& s : d.data.imu) {
    if (s.t < t_first - 0.1 || s.t > t_first + 0.1) imu.push_back(s);
  }
  const DynamicInitResult r = run(d, imu, t_first);
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.reason, "imu_gap");
}

TEST(DynamicInit, RejectsNonPositiveGrowth) {
  const Data d = handheld();
  DynamicInitConfig cfg;
  cfg.growth = 0.0;
  EXPECT_THROW(run(d, d.data.imu, 1.0, cfg), std::invalid_argument);
}

// Three keyframes where the first and last share no feature: two pairs, four
// equations for six unknowns. Refused ("pairs") instead of reading a sixth
// singular value that does not exist.
TEST(FeaturelessInit, RefusesFewerEquationsThanUnknowns) {
  const Data d = handheld();
  std::vector<InitFrame> kf{d.frames[20], d.frames[30], d.frames[40]};
  const auto keep = [](InitFrame& f, int rem) {
    auto& b = f.bearings;
    b.erase(std::remove_if(b.begin(), b.end(), [rem](const auto& x) { return x.first % 2 != rem; }), b.end());
  };
  keep(kf[0], 0);
  keep(kf[2], 1);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const FeaturelessInitResult r = featurelessInit(d.data.imu, kf, T_IC(d.c), zero, zero, d.c.imu_noise);
  EXPECT_EQ(r.pairs, 2);
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.reason, "pairs");
}

}  // namespace
}  // namespace msckf
