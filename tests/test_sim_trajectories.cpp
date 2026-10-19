// Simulator trajectories beyond the spline (time warp, analytic Lissajous) and
// the simulator's random streams and turn-on biases (user's simulator plan).

#include <gtest/gtest.h>

#include "msckf/core/constants.h"
#include "msckf/sim/lissajous.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"
#include "msckf/sim/time_warp.h"
#include "trajectory_checks.h"

#include <cmath>

namespace msckf {
namespace {

constexpr double kStatic = 2.0, kRamp = 1.5;

TrajectoryPtr figure8(const LissajousConfig& config = {}) {
  return std::make_shared<LissajousTrajectory>(config, 30.0);
}

TEST(TimeWarp, HoldsFirstPoseWithZeroRatesAtRest) {
  const TrajectoryPtr base = figure8();
  const TimeWarpedTrajectory warped(base, kStatic, kRamp);
  const TrajectoryState first = base->evaluate(0.0);
  for (double t = 0.0; t <= kStatic; t += 0.25) {
    const TrajectoryState s = warped.evaluate(t);
    EXPECT_EQ(s.p_GI, first.p_GI);
    EXPECT_LT((s.R_GI.inverse() * first.R_GI).log().norm(), 1e-15);
    EXPECT_EQ(s.v_GI.norm() + s.a_GI.norm() + s.omega_I.norm() + s.alpha_I.norm(), 0.0);
  }
}

TEST(TimeWarp, FollowsTheBaseAtFullSpeedAfterTheRamp) {
  const TrajectoryPtr base = figure8();
  const TimeWarpedTrajectory warped(base, kStatic, kRamp);
  for (double u = 0.0; u < 10.0; u += 0.7) {
    const TrajectoryState s = warped.evaluate(kStatic + kRamp + u), b = base->evaluate(kRamp / 2.0 + u);
    EXPECT_LT((s.p_GI - b.p_GI).norm() + (s.v_GI - b.v_GI).norm() + (s.omega_I - b.omega_I).norm(), 1e-12);
  }
  EXPECT_DOUBLE_EQ(warped.maxTime(), base->maxTime() + kStatic + kRamp / 2.0);
}

TEST(TimeWarp, RejectsStaticStartWithoutRamp) {
  EXPECT_THROW(TimeWarpedTrajectory(figure8(), kStatic, 0.0), std::invalid_argument);
}

TEST(Lissajous, TiltedTangentFigureHasExactDerivatives) {
  LissajousConfig c;
  c.heading = Heading::kTangent;
  c.amplitude.z() = 1.0;
  c.R_WF = Sophus::SO3d::rotX(0.4) * Sophus::SO3d::rotZ(0.3);
  const DerivativeErrors e = derivativeErrors(*figure8(c));
  EXPECT_LT(e.v, 1e-7);
  EXPECT_LT(e.a, 1e-7);
  EXPECT_LT(e.omega, 1e-7);
  EXPECT_LT(e.alpha, 1e-6);
}

TEST(Lissajous, FigureAttitudeChangesOnlyTheAccelerometer) {
  LissajousConfig tilted;
  tilted.R_WF = Sophus::SO3d::rotY(M_PI / 2.0);  // vertical figure-8
  const TrajectoryPtr flat = figure8(), upright = figure8(tilted);
  double max_gyro_diff = 0.0, min_acc_diff = 1e9;
  for (double t = 0.0; t < 20.0; t += 0.5) {
    const TrajectoryState a = flat->evaluate(t), b = upright->evaluate(t);
    max_gyro_diff = std::max(max_gyro_diff, (a.omega_I - b.omega_I).norm());
    const Eigen::Vector3d f_a = a.R_GI.inverse() * (a.a_GI - kGravity), f_b = b.R_GI.inverse() * (b.a_GI - kGravity);
    min_acc_diff = std::min(min_acc_diff, (f_a - f_b).norm());
  }
  EXPECT_EQ(max_gyro_diff, 0.0);
  EXPECT_GT(min_acc_diff, 1.0);  // gravity turned by 90 degrees in the body
}

TEST(Lissajous, PlanarHeadingPointsAlongTheHorizontalVelocity) {
  LissajousConfig c;
  c.roll = c.pitch = c.yaw = AngleWave{};
  const TrajectoryPtr traj = figure8(c);
  for (double t = 0.0; t < 20.0; t += 0.3) {
    const TrajectoryState s = traj->evaluate(t);
    const Eigen::Vector3d forward = s.R_GI * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(forward.dot(Eigen::Vector3d(s.v_GI.x(), s.v_GI.y(), 0.0).normalized()), 1.0, 1e-12);
  }
}

TEST(Lissajous, TangentHeadingPointsAlongTheVelocity) {  // pitch sign: climbing tilts the nose up
  LissajousConfig c;
  c.heading = Heading::kTangent;
  c.amplitude.z() = 1.0;
  c.roll = c.pitch = c.yaw = AngleWave{};
  const TrajectoryPtr traj = figure8(c);
  for (double t = 0.0; t < 20.0; t += 0.3) {
    const TrajectoryState s = traj->evaluate(t);
    EXPECT_NEAR((s.R_GI * Eigen::Vector3d::UnitX()).dot(s.v_GI.normalized()), 1.0, 1e-12);
  }
}

TEST(Lissajous, RejectsPathFollowingWhereThePathStops) {
  LissajousConfig c;
  c.amplitude = Eigen::Vector3d(0.0, 0.0, 1.0);
  EXPECT_THROW(LissajousTrajectory(c, 10.0), std::invalid_argument);
}

SimConfig shortRun(uint64_t seed) {
  SimConfig c = defaultSimConfig(Scenario::kFigure8);
  c.duration = 1.0;
  c.seed = seed;
  return c;
}

SimData simulate(const SimConfig& c) {
  return Simulator(c, makeScenario(Scenario::kFigure8, 2.0, 1), simulatedCamera()).run();
}

TEST(SimulatorStreams, LandmarkDensityLeavesImuNoiseUnchanged) {
  SimConfig sparse = shortRun(4), dense = sparse;
  dense.wall_density *= 2.0;
  const SimData a = simulate(sparse), b = simulate(dense);
  ASSERT_NE(a.landmarks.size(), b.landmarks.size());
  for (size_t k = 0; k < a.imu.size(); ++k) {
    ASSERT_EQ(a.imu[k].gyro, b.imu[k].gyro);
    ASSERT_EQ(a.imu[k].acc, b.imu[k].acc);
  }
}

TEST(SimulatorStreams, ImuRateLeavesLandmarksUnchanged) {
  SimConfig slow = shortRun(4), fast = slow;
  fast.imu_rate *= 2.0;
  const SimData a = simulate(slow), b = simulate(fast);
  ASSERT_EQ(a.landmarks.size(), b.landmarks.size());
  for (size_t i = 0; i < a.landmarks.size(); ++i) ASSERT_EQ(a.landmarks[i].p_G, b.landmarks[i].p_G);
}

TEST(SimulatorStreams, SameSeedIsBitIdentical) {
  const SimData a = simulate(shortRun(9)), b = simulate(shortRun(9));
  for (size_t k = 0; k < a.imu.size(); ++k) ASSERT_EQ(a.imu[k].acc, b.imu[k].acc);
  ASSERT_EQ(a.frames.back().observations.size(), b.frames.back().observations.size());
  for (size_t i = 0; i < a.frames.back().observations.size(); ++i) {
    ASSERT_EQ(a.frames.back().observations[i].uv, b.frames.back().observations[i].uv);
  }
}

TEST(TurnOnBias, DrawnPerRunWithTheConfiguredSigma) {
  constexpr int runs = 400;
  double ss_g = 0.0, ss_a = 0.0;
  SimConfig c = shortRun(0);
  c.duration = 0.0;
  c.ground_density = c.wall_density = c.ceiling_density = 0.0;
  for (int r = 0; r < runs; ++r) {
    c.seed = static_cast<uint64_t>(r);
    const GroundTruth& g0 = simulate(c).imu_truth.front();
    ss_g += g0.gyro_bias.squaredNorm();
    ss_a += g0.acc_bias.squaredNorm();
  }
  EXPECT_NEAR(std::sqrt(ss_g / (3.0 * runs)) / c.gyro_turn_on_sigma, 1.0, 0.1);
  EXPECT_NEAR(std::sqrt(ss_a / (3.0 * runs)) / c.acc_turn_on_sigma, 1.0, 0.1);
}

}  // namespace
}  // namespace msckf
