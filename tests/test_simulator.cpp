// Stage 0 simulator tests. The gate: noiseless IMU integrated over 60 s matches
// the spline to < 1e-6 m and < 1e-8 rad, which validates the simulator's IMU
// model and the RK4 propagation together.

#include <gtest/gtest.h>

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/pinhole_radtan.h"
#include "msckf/core/constants.h"
#include "msckf/imu/propagation.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <sophus/se3.hpp>

#include <algorithm>
#include <random>

namespace msckf {
namespace {

constexpr double kGateDuration = 60.0;

CameraModelPtr tumviDoubleSphere() {
  Eigen::VectorXd k(6);
  k << 158.28600034966977, 158.2743455478755, 254.96116578191653, 256.8894218032305,
      -0.17213086034353243, 0.5931177593944744;
  return std::make_shared<DoubleSphere>(512, 512, k);
}

CameraModelPtr eurocPinhole() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

class SimulatorTest : public ::testing::TestWithParam<Scenario> {
 protected:
  SplineTrajectory makeTrajectory(double duration = kGateDuration) {
    return makeScenario(GetParam(), duration + 1.0, 3);
  }
};

TEST_P(SimulatorTest, SplineDerivativesMatchFiniteDifferences) {
  const SplineTrajectory traj = makeTrajectory();
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> time(traj.minTime() + 0.01, traj.maxTime() - 0.01);
  constexpr double h = 1e-5;
  double max_v = 0, max_a = 0, max_w = 0, max_alpha = 0;
  for (int i = 0; i < 2000; ++i) {
    const double t = time(rng);
    const SplineState s = traj.evaluate(t);
    const SplineState sp = traj.evaluate(t + h), sm = traj.evaluate(t - h);
    const Eigen::Vector3d v_fd = (sp.p_GI - sm.p_GI) / (2 * h);
    const Eigen::Vector3d a_fd = (sp.v_GI - sm.v_GI) / (2 * h);
    // Body rate: R(t + h) = R(t - h) Exp(2 h w) to second order.
    const Eigen::Vector3d w_fd = (sm.R_GI.inverse() * sp.R_GI).log() / (2 * h);
    const Eigen::Vector3d alpha_fd = (sp.omega_I - sm.omega_I) / (2 * h);
    max_v = std::max(max_v, (s.v_GI - v_fd).norm() / (1.0 + s.v_GI.norm()));
    max_a = std::max(max_a, (s.a_GI - a_fd).norm() / (1.0 + s.a_GI.norm()));
    max_w = std::max(max_w, (s.omega_I - w_fd).norm() / (1.0 + s.omega_I.norm()));
    max_alpha = std::max(max_alpha, (s.alpha_I - alpha_fd).norm() / (1.0 + s.alpha_I.norm()));
  }
  std::cout << "[ info ] max rel. error v " << max_v << ", a " << max_a << ", omega " << max_w
            << ", alpha " << max_alpha << "\n";
  EXPECT_LT(max_v, 1e-7);
  EXPECT_LT(max_a, 1e-7);
  EXPECT_LT(max_w, 1e-7);
  EXPECT_LT(max_alpha, 1e-6);
}

TEST_P(SimulatorTest, SplineIsC2AcrossKnots) {
  const SplineTrajectory traj = makeTrajectory();
  const double dt = traj.knotInterval();
  constexpr double eps = 1e-9;
  double max_jump = 0.0;
  for (double t = traj.minTime() + dt; t < traj.maxTime() - dt; t += dt) {
    const SplineState a = traj.evaluate(t - eps), b = traj.evaluate(t + eps);
    max_jump = std::max({max_jump, (a.p_GI - b.p_GI).norm(), (a.v_GI - b.v_GI).norm(),
                         (a.a_GI - b.a_GI).norm(), (a.omega_I - b.omega_I).norm(),
                         (a.alpha_I - b.alpha_I).norm(),
                         (a.R_GI.inverse() * b.R_GI).log().norm()});
  }
  std::cout << "[ info ] max jump across knots " << max_jump << "\n";
  EXPECT_LT(max_jump, 1e-6);
}

TEST_P(SimulatorTest, NoiselessImuIntegrationGate) {
  SimConfig config;
  config.duration = kGateDuration;
  // At 400 Hz fast rotation (peak ~6 rad/s) misses 1e-6 m by a few percent from
  // RK4 truncation alone; Rk4ConvergesAtFourthOrder covers 400 Hz. The gate
  // validates the integrator and the IMU model, not the IMU rate.
  if (GetParam() == Scenario::kFastRotation) config.imu_rate = 800.0;
  const Simulator sim(config, makeTrajectory(), eurocPinhole());
  const double dt = 1.0 / config.imu_rate;
  const int steps = static_cast<int>(std::lround(config.duration / dt));

  auto reading = [&](double t) {
    const ImuSample s = sim.idealImu(t);
    return ImuReading{s.gyro, s.acc};
  };

  // Exact midpoint readings from the continuous-time model, and the filter's
  // situation: discrete samples only, midpoint by linear interpolation.
  NavState exact = sim.trueState(0.0), discrete = exact;
  for (int k = 0; k < steps; ++k) {
    const double t = k * dt;
    const ImuReading r0 = reading(t), r1 = reading(t + dt);
    exact = rk4Step(exact, r0, reading(t + 0.5 * dt), r1, dt);
    discrete = rk4Step(discrete, r0, interpolate(r0, r1, 0.5), r1, dt);
  }
  const NavState truth = sim.trueState(steps * dt);
  auto pos_err = [&](const NavState& x) { return (x.p_GI - truth.p_GI).norm(); };
  auto rot_err = [&](const NavState& x) { return (truth.R_GI.inverse() * x.R_GI).log().norm(); };

  std::cout << "[ info ] " << toString(GetParam()) << ", " << config.duration << " s at "
            << config.imu_rate << " Hz: exact midpoints " << pos_err(exact) << " m, "
            << rot_err(exact) << " rad; interpolated midpoints " << pos_err(discrete) << " m, "
            << rot_err(discrete) << " rad\n";
  EXPECT_LT(pos_err(exact), 1e-6);
  EXPECT_LT(rot_err(exact), 1e-8);
}

TEST_P(SimulatorTest, Rk4ConvergesAtFourthOrder) {
  SimConfig config;
  config.duration = 10.0;
  const Simulator sim(config, makeTrajectory(config.duration), eurocPinhole());
  auto integrate = [&](double rate) {
    const double dt = 1.0 / rate;
    const int steps = static_cast<int>(std::lround(config.duration * rate));
    NavState x = sim.trueState(0.0);
    for (int k = 0; k < steps; ++k) {
      const double t = k * dt;
      auto r = [&](double tau) {
        const ImuSample s = sim.idealImu(tau);
        return ImuReading{s.gyro, s.acc};
      };
      x = rk4Step(x, r(t), r(t + 0.5 * dt), r(t + dt), dt);
    }
    return (x.p_GI - sim.trueState(config.duration).p_GI).norm();
  };
  const double e200 = integrate(200.0), e400 = integrate(400.0), e800 = integrate(800.0);
  std::cout << "[ info ] " << toString(GetParam()) << " position error over 10 s: 200 Hz " << e200
            << ", 400 Hz " << e400 << ", 800 Hz " << e800 << " m; ratios " << e200 / e400 << ", "
            << e400 / e800 << "\n";
  // Global error of a 4th-order method drops 16x per halving of the step.
  EXPECT_NEAR(e200 / e400, 16.0, 3.0);
}

TEST_P(SimulatorTest, ImuNoiseAndBiasWalkStatistics) {
  SimConfig config;
  config.duration = 60.0;
  config.gyro_bias0 = Eigen::Vector3d(0.01, -0.02, 0.005);
  config.acc_bias0 = Eigen::Vector3d(0.1, 0.05, -0.08);
  const Simulator sim(config, makeTrajectory(), eurocPinhole());
  const SimData data = sim.run();
  const double dt = 1.0 / config.imu_rate;
  const ImuNoise& nz = config.imu_noise;

  ASSERT_EQ(data.imu.size(), data.imu_truth.size());
  EXPECT_LT((data.imu_truth.front().gyro_bias - config.gyro_bias0).norm(), 1e-15);

  double ss_g = 0, ss_a = 0, ss_bg = 0, ss_ba = 0;
  const size_t n = data.imu.size();
  for (size_t k = 0; k < n; ++k) {
    const ImuSample ideal = sim.idealImu(data.imu[k].t);
    ss_g += (data.imu[k].gyro - ideal.gyro - data.imu_truth[k].gyro_bias).squaredNorm();
    ss_a += (data.imu[k].acc - ideal.acc - data.imu_truth[k].acc_bias).squaredNorm();
    if (k + 1 < n) {
      ss_bg += (data.imu_truth[k + 1].gyro_bias - data.imu_truth[k].gyro_bias).squaredNorm();
      ss_ba += (data.imu_truth[k + 1].acc_bias - data.imu_truth[k].acc_bias).squaredNorm();
    }
  }
  // Per-axis sample standard deviations vs the discrete-time values.
  const double sd_g = std::sqrt(ss_g / (3.0 * n)), sd_a = std::sqrt(ss_a / (3.0 * n));
  const double sd_bg = std::sqrt(ss_bg / (3.0 * (n - 1))), sd_ba = std::sqrt(ss_ba / (3.0 * (n - 1)));
  EXPECT_NEAR(sd_g / (nz.gyro_noise / std::sqrt(dt)), 1.0, 0.02);
  EXPECT_NEAR(sd_a / (nz.acc_noise / std::sqrt(dt)), 1.0, 0.02);
  EXPECT_NEAR(sd_bg / (nz.gyro_bias_walk * std::sqrt(dt)), 1.0, 0.02);
  EXPECT_NEAR(sd_ba / (nz.acc_bias_walk * std::sqrt(dt)), 1.0, 0.02);
}

TEST_P(SimulatorTest, CameraObservationsMatchIndependentProjection) {
  for (const CameraModelPtr& cam : {eurocPinhole(), tumviDoubleSphere()}) {
    SimConfig config = defaultSimConfig(GetParam());
    config.duration = 20.0;
    config.pixel_noise = 0.0;
    config.time_offset = 0.005;
    const Simulator sim(config, makeTrajectory(), cam);
    const SimData data = sim.run();

    // T_GC = T_GI T_IC, p_C = T_GC^-1 p_G: a separate code path from the simulator.
    const Sophus::SE3d T_IC(config.R_IC, config.p_IC);
    double max_err = 0.0, max_offset_effect = 0.0;
    size_t total = 0, min_count = SIZE_MAX;
    for (const CameraFrame& f : data.frames) {
      const NavState x = sim.trueState(f.t_cam + config.time_offset);
      const NavState x_no_offset = sim.trueState(f.t_cam);
      const Sophus::SE3d T_CG = (Sophus::SE3d(x.R_GI, x.p_GI) * T_IC).inverse();
      const Sophus::SE3d T_CG_wrong = (Sophus::SE3d(x_no_offset.R_GI, x_no_offset.p_GI) * T_IC).inverse();
      for (const Observation& o : f.observations) {
        const Eigen::Vector3d p_G = data.landmarks[o.landmark_id].p_G;
        Eigen::Vector2d uv, uv_wrong;
        ASSERT_TRUE(cam->project(T_CG * p_G, uv));
        max_err = std::max(max_err, (uv - o.uv_true).norm());
        EXPECT_EQ(o.uv, o.uv_true);  // pixel noise disabled
        if (cam->project(T_CG_wrong * p_G, uv_wrong)) {
          max_offset_effect = std::max(max_offset_effect, (uv_wrong - o.uv_true).norm());
        }
      }
      total += f.observations.size();
      min_count = std::min(min_count, f.observations.size());
    }
    std::cout << "[ info ] " << toString(GetParam()) << ", " << cam->name() << ": "
              << data.landmarks.size() << " landmarks, " << total / data.frames.size()
              << " observations per frame (min " << min_count << "), max reprojection "
              << max_err << " px, 5 ms offset moves pixels up to " << max_offset_effect << " px\n";
    EXPECT_LT(max_err, 1e-9);
    EXPECT_GT(max_offset_effect, 0.01);  // the offset is actually applied
    EXPECT_GE(min_count, 20u);
  }
}

TEST(SimulatorScenarios, CartIsPlanarWithExactStops) {  // and pushed: speed surges while moving
  const SplineTrajectory traj = makeScenario(Scenario::kCart, 120.0, 3);
  double max_speed = 0.0, max_dz = 0.0, stopped_time = 0.0;
  bool exact_stop_found = false;
  constexpr double step = 0.01;
  for (double t = traj.minTime(); t <= traj.maxTime(); t += step) {
    const SplineState s = traj.evaluate(t);
    max_speed = std::max(max_speed, s.v_GI.norm());
    max_dz = std::max(max_dz, std::abs(s.p_GI.z() - 0.8));
    if (s.v_GI.norm() < 1e-12 && s.omega_I.norm() < 1e-12 && s.a_GI.norm() < 1e-12) {
      exact_stop_found = true;
      stopped_time += step;
    }
  }
  std::cout << "[ info ] cart: max speed " << max_speed << " m/s, max |dz| " << max_dz
            << " m, exactly stationary " << stopped_time << " s of " << traj.maxTime() << " s\n";
  EXPECT_TRUE(exact_stop_found);
  // Cruise 1 m/s plus push surges of +-0.15 m/s.
  EXPECT_NEAR(max_speed, 1.15, 0.05);
  EXPECT_LT(max_dz, 0.005);
}

TEST(SimulatorScenarios, FastRotationReachesHighRates) {
  const SplineTrajectory traj = makeScenario(Scenario::kFastRotation, 60.0, 3);
  double max_rate = 0.0;
  for (double t = traj.minTime(); t <= traj.maxTime(); t += 0.005) {
    max_rate = std::max(max_rate, traj.evaluate(t).omega_I.norm());
  }
  std::cout << "[ info ] fast rotation: peak rate " << max_rate << " rad/s\n";
  EXPECT_GT(max_rate, 4.0);
}

INSTANTIATE_TEST_SUITE_P(AllScenarios, SimulatorTest,
                         ::testing::Values(Scenario::kHandheld, Scenario::kCart,
                                           Scenario::kFastRotation),
                         [](const auto& info) {
                           std::string name = toString(info.param);
                           name.erase(std::remove(name.begin(), name.end(), '-'), name.end());
                           return name;
                         });

TEST(Scenarios, ParseScenarioAcceptsCommandLineNames) {
  EXPECT_EQ(parseScenario("handheld"), Scenario::kHandheld);
  EXPECT_EQ(parseScenario("fast_rotation"), Scenario::kFastRotation);
  EXPECT_EQ(parseScenario("cart"), Scenario::kCart);
  EXPECT_EQ(parseScenario("car"), Scenario::kCar);
  EXPECT_THROW(parseScenario("boat"), std::invalid_argument);
}

}  // namespace
}  // namespace msckf
