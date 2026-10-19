// Stage 1b: dynamic initialization. Step 1, the feature-less linear solve
// (sqrtVINS Sec. V-A): exact on noise-free simulation, within the plan's
// accuracy targets with noise, refuses a window without acceleration. Step 2,
// the iterated EKF refinement: converges to the truth from a perturbed start
// without noise, improves short windows, and hands off a consistent
// covariance (NEES of roll, pitch, velocity and gyro bias in the 95% band).

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/core/constants.h"
#include "msckf/eval/nees.h"
#include "msckf/init/dynamic_init.h"
#include "msckf/init/featureless_init.h"
#include "msckf/init/init_refinement.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"
#include "msckf/sim/spline.h"

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <cmath>
#include <vector>

namespace msckf {
namespace {

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

struct Window {
  std::vector<InitFrame> frames;
  Eigen::Vector3d v_I0, g_I0;  // truth
};

// The frames of [t0, t0 + length], bearings from the noisy (or exact) pixels.
Window makeWindow(const Simulator& sim, const SimData& data, double t0, double length, bool exact_pixels) {
  Window w;
  for (const CameraFrame& f : data.frames) {
    const double t = f.t_cam + sim.config().time_offset;
    if (t < t0 - 1e-9 || t > t0 + length + 1e-9) continue;
    InitFrame fr;
    fr.t = t;
    for (const Observation& o : f.observations) {
      Eigen::Vector3d b;
      if (sim.camera().unproject(exact_pixels ? o.uv_true : o.uv, b)) fr.bearings.emplace_back(o.landmark_id, b);
    }
    w.frames.push_back(fr);
  }
  const NavState s = sim.trueState(w.frames.front().t);
  w.v_I0 = s.R_GI.inverse() * s.v_GI;
  w.g_I0 = s.R_GI.inverse() * kGravity;
  return w;
}

double angleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b)) * 180.0 / M_PI;
}

Sophus::SE3d T_IC(const SimConfig& c) { return Sophus::SE3d(c.R_IC, c.p_IC); }

TEST(FeaturelessInit, ExactWithoutNoise) {
  SimConfig c = defaultSimConfig(Scenario::kHandheld);
  c.duration = 12.0;
  c.imu_rate = 2000.0;  // preintegration's own (second-order) error out of the way
  c.imu_noise_enabled = false;
  c.pixel_noise = 0.0;
  const Simulator sim(c, makeScenario(Scenario::kHandheld, 13.0, 5), camera());
  const SimData data = sim.run();
  for (double t0 : {1.0, 4.0, 8.0}) {
    const Window w = makeWindow(sim, data, t0, 1.0, true);
    const FeaturelessInitResult r =
        featurelessInit(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), c.imu_noise);
    ASSERT_TRUE(r.ok) << r.reason << " at t0 " << t0;
    EXPECT_LT(angleDeg(r.g_I0, w.g_I0), 1e-3) << t0;
    EXPECT_LT((r.v_I0 - w.v_I0).norm(), 1e-4) << t0;
    EXPECT_NEAR(r.g_I0.norm(), 9.81, 1e-12);
  }
}

TEST(FeaturelessInit, MeetsAccuracyTargetsWithNoise) {
  // Plan, Stage 1 metrics: gravity direction < 1 deg, velocity < 0.1 m/s or
  // 10 %. Medians over 1 s windows at many starts and seeds, biases known.
  std::vector<double> grav, vel;
  int ok = 0, total = 0;
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    SimConfig c = defaultSimConfig(Scenario::kHandheld);
    c.duration = 21.0;
    c.seed = seed;
    const Simulator sim(c, makeScenario(Scenario::kHandheld, 22.0, seed), camera());
    const SimData data = sim.run();
    for (double t0 = 1.0; t0 + 1.0 < c.duration; t0 += 2.0) {
      const Window w = makeWindow(sim, data, t0, 1.0, false);
      const FeaturelessInitResult r = featurelessInit(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero(),
                                                      Eigen::Vector3d::Zero(), c.imu_noise);
      ++total;
      if (!r.ok) continue;
      ++ok;
      grav.push_back(angleDeg(r.g_I0, w.g_I0));
      vel.push_back((r.v_I0 - w.v_I0).norm());
    }
  }
  ASSERT_FALSE(grav.empty());
  std::sort(grav.begin(), grav.end());
  std::sort(vel.begin(), vel.end());
  const double g50 = grav[grav.size() / 2], g90 = grav[grav.size() * 9 / 10];
  const double v50 = vel[vel.size() / 2], v90 = vel[vel.size() * 9 / 10];
  std::cout << "[ info ] handheld, 1 s windows: accepted " << ok << "/" << total << "; gravity error median " << g50
            << " deg, p90 " << g90 << " deg; velocity error median " << v50 << " m/s, p90 " << v90 << " m/s\n";
  EXPECT_GE(ok, total * 9 / 10);
  EXPECT_LT(g50, 1.0);
  EXPECT_LT(v50, 0.1);
}

TEST(FeaturelessInit, SelectsKeyframesByParallax) {
  SimConfig c = defaultSimConfig(Scenario::kHandheld);
  c.duration = 4.0;
  const Simulator sim(c, makeScenario(Scenario::kHandheld, 5.0, 2), camera());
  const SimData data = sim.run();
  // 0.1 s at 20 Hz: 3 frames, all keyframes (as in the paper).
  const Window short_w = makeWindow(sim, data, 1.0, 0.1, false);
  EXPECT_EQ(selectKeyframes(data.imu, short_w.frames, T_IC(c), Eigen::Vector3d::Zero()).size(), 3u);
  // 1 s: 21 frames, 5 keyframes from first to last, in order.
  const Window w = makeWindow(sim, data, 1.0, 1.0, false);
  const std::vector<int> kf = selectKeyframes(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero());
  ASSERT_EQ(kf.size(), 5u);
  EXPECT_EQ(kf.front(), 0);
  EXPECT_EQ(kf.back(), static_cast<int>(w.frames.size()) - 1);
  EXPECT_TRUE(std::is_sorted(kf.begin(), kf.end()));
  EXPECT_EQ(std::adjacent_find(kf.begin(), kf.end()), kf.end());
}

TEST(FeaturelessInit, RefusesWindowWithoutAcceleration) {
  // Straight line at constant velocity and attitude: scale is unobservable.
  SimConfig c = defaultSimConfig(Scenario::kHandheld);
  c.duration = 3.0;
  constexpr double kKnotDt = 0.1;
  SplineTrajectory spline(0.0, kKnotDt);
  for (int k = 0; k < 40; ++k) {
    spline.addControlPose(Sophus::SO3d::rotZ(0.3), Eigen::Vector3d(0.5 * k * kKnotDt, 0.0, 1.5));
  }
  const Simulator sim(c, std::make_shared<SplineTrajectory>(spline), camera());
  const SimData data = sim.run();
  const Window w = makeWindow(sim, data, 1.0, 1.0, false);
  const FeaturelessInitResult r =
      featurelessInit(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), c.imu_noise);
  std::cout << "[ info ] constant velocity: acc excitation " << r.acc_excitation << " m/s^2, singular value "
            << r.singular_value << ", reason " << r.reason << "\n";
  EXPECT_FALSE(r.ok);
}

// The initializer's frame in the simulator's world: gravity aligned, origin
// at the first frame's IMU position, yaw aligned to the estimate's first
// clone. Yaw is the gauge: the initializer fixes it from its own first
// estimate of "up", so a tilt error at the start rotates its frame about the
// vertical, which is not an error.
Sophus::SE3d initFromWorld(const Simulator& sim, double t_first, const Sophus::SO3d& R_GI0_est) {
  const NavState s0 = sim.trueState(t_first);
  Sophus::SO3d R = attitudeFromUp(-(s0.R_GI.inverse() * kGravity).normalized()) * s0.R_GI.inverse();
  const Eigen::Matrix3d D = (R_GI0_est * (R * s0.R_GI).inverse()).matrix();  // ~ a rotation about z
  R = Sophus::SO3d::rotZ(std::atan2(D(1, 0) - D(0, 1), D(0, 0) + D(1, 1))) * R;
  return Sophus::SE3d(R, -(R * s0.p_GI));
}

NavState truthInInitFrame(const Simulator& sim, double t_first, double t, const Sophus::SO3d& R_GI0_est) {
  const Sophus::SE3d T = initFromWorld(sim, t_first, R_GI0_est);
  const NavState s = sim.trueState(t);
  NavState out;
  out.R_GI = T.so3() * s.R_GI;
  out.p_GI = T * s.p_GI;
  out.v_GI = T.so3() * s.v_GI;
  return out;
}

TEST(InitRefinement, ConvergesToTruthFromPerturbedStartWithoutNoise) {
  SimConfig c = defaultSimConfig(Scenario::kHandheld);
  c.duration = 6.0;
  c.imu_rate = 2000.0;
  c.imu_noise_enabled = false;
  c.pixel_noise = 0.0;
  const Simulator sim(c, makeScenario(Scenario::kHandheld, 7.0, 5), camera());
  const SimData data = sim.run();
  const Window w = makeWindow(sim, data, 2.0, 0.5, true);
  FeaturelessInitResult lin;  // the truth, perturbed: 5 deg of tilt, 0.3 m/s
  lin.g_I0 = Sophus::SO3d::rotX(5.0 * M_PI / 180.0) * w.g_I0;
  lin.v_I0 = w.v_I0 + Eigen::Vector3d(0.3, -0.2, 0.1);
  InitRefinementConfig rc;
  rc.bearing_sigma = 1e-4;
  // The start is wrong on purpose, so its priors must not pull: uninformative.
  rc.tilt_sigma = 10.0;
  rc.velocity_sigma = 100.0;
  const InitRefinementResult r = refineInit(data.imu, w.frames, T_IC(c), lin, Eigen::Vector3d::Zero(),
                                            Eigen::Vector3d::Zero(), c.imu_noise, rc);
  ASSERT_TRUE(r.ok) << r.reason;
  const NavState truth = truthInInitFrame(sim, w.frames.front().t, w.frames.back().t, r.clones.front().R_GI);
  const Eigen::Vector3d up_est = r.state.nav.R_GI.inverse() * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d up_true = truth.R_GI.inverse() * Eigen::Vector3d::UnitZ();
  std::cout << "[ info ] noise-free from 5 deg / 0.37 m/s off: " << r.iterations << " iterations, tilt error "
            << angleDeg(up_est, up_true) << " deg, velocity error " << (r.state.nav.v_GI - truth.v_GI).norm()
            << " m/s\n";
  EXPECT_LT(angleDeg(up_est, up_true), 0.01);
  EXPECT_LT((r.state.nav.v_GI - truth.v_GI).norm(), 1e-3);
  // Landmarks in the state land on the true points.
  const Sophus::SE3d T = initFromWorld(sim, w.frames.front().t, r.clones.front().R_GI);
  ASSERT_GT(r.landmarks.size(), 20u);
  double max_err = 0.0;
  for (const auto& lm : r.landmarks) max_err = std::max(max_err, (lm.p_G - T * data.landmarks.at(lm.id).p_G).norm());
  std::cout << "[ info ] " << r.landmarks.size() << " landmarks, max position error " << max_err << " m\n";
  EXPECT_LT(max_err, 1e-3);
  EXPECT_EQ(r.P_joint.rows(), 15 + 3 * static_cast<int>(r.landmarks.size()));
}

TEST(InitRefinement, ImprovesShortWindowsAndIsConsistent) {
  // Monte Carlo: 0.3 s windows (where the linear solve alone is poor) and 1 s
  // windows. NEES of [tilt (2), velocity (3), gyro bias (3)] at the hand-off,
  // with the true biases drawn from the refinement's bias prior (a fair NEES:
  // gyro bias is barely observable in a window, so its estimate stays near the
  // prior). The linear solve, which assumes known biases, is given the prior
  // mean (zero), as on real data.
  const InitRefinementConfig defaults;
  for (double length : {0.3, 0.5, 1.0}) {
    std::vector<double> lin_err, ref_err, nees, lm_nees;
    std::map<std::string, int> refused;
    for (uint64_t seed = 1; seed <= 10; ++seed) {
      SimConfig c = defaultSimConfig(Scenario::kHandheld);
      c.duration = 21.0;
      c.seed = seed;
      std::mt19937_64 rng(100 + seed);
      std::normal_distribution<double> nd(0.0, 1.0);
      c.gyro_bias0 = defaults.gyro_bias_sigma * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
      c.acc_bias0 = defaults.acc_bias_sigma * Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
      const Simulator sim(c, makeScenario(Scenario::kHandheld, 22.0, seed), camera());
      const SimData data = sim.run();
      InitRefinementConfig rc;
      rc.bearing_sigma = c.pixel_noise / 458.0;  // 1 px at EuRoC's focal length
      for (double t0 = 1.0; t0 + length < c.duration; t0 += 2.0) {
        Window w = makeWindow(sim, data, t0, length, false);
        std::vector<InitFrame> kf;  // the pipeline: keyframes by parallax
        for (int k : selectKeyframes(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero())) kf.push_back(w.frames[k]);
        w.frames = kf;
        const FeaturelessInitResult lin = featurelessInit(data.imu, w.frames, T_IC(c), Eigen::Vector3d::Zero(),
                                                          Eigen::Vector3d::Zero(), c.imu_noise);
        if (!lin.ok) continue;
        const InitRefinementResult r = refineInit(data.imu, w.frames, T_IC(c), lin, Eigen::Vector3d::Zero(),
                                                  Eigen::Vector3d::Zero(), c.imu_noise, rc);
        if (!r.ok) {
          ++refused[r.reason];
          continue;
        }
        const NavState truth =
            truthInInitFrame(sim, w.frames.front().t, w.frames.back().t, r.clones.front().R_GI);
        lin_err.push_back(angleDeg(lin.g_I0, w.g_I0));
        const Eigen::Vector3d g_ref = r.clones.front().R_GI.inverse() * kGravity;
        ref_err.push_back(angleDeg(g_ref, w.g_I0));
        // Error vector: tilt as the horizontal components of the global-frame
        // rotation error, velocity, gyro bias (truth: the simulator's, 0).
        const Eigen::Matrix3d R = r.state.nav.R_GI.matrix();
        const Eigen::Vector3d dtheta_G = R * (r.state.nav.R_GI.inverse() * truth.R_GI).log();
        Eigen::Matrix<double, 8, 1> e;
        // Bias truth at the hand-off: the simulator's walk, at the last frame.
        Eigen::Vector3d bg_true = c.gyro_bias0;
        for (const GroundTruth& gtk : data.imu_truth) {
          if (gtk.t >= w.frames.back().t - 1e-9) {
            bg_true = gtk.gyro_bias;
            break;
          }
        }
        e << dtheta_G.head<2>(), truth.v_GI - r.state.nav.v_GI, bg_true - r.state.gyro_bias;
        Eigen::Matrix<double, 8, 15> S = Eigen::Matrix<double, 8, 15>::Zero();
        S.block<2, 3>(0, imu_err::kTheta) = R.topRows<2>();
        S.block<3, 3>(2, imu_err::kV) = Eigen::Matrix3d::Identity();
        S.block<3, 3>(5, imu_err::kBg) = Eigen::Matrix3d::Identity();
        const Eigen::Matrix<double, 8, 8> Pe = S * r.state.P * S.transpose();
        nees.push_back(e.dot(Pe.ldlt().solve(e)));
        // One landmark per run (runs independent): its position NEES.
        if (!r.landmarks.empty()) {
          const Sophus::SE3d T = initFromWorld(sim, w.frames.front().t, r.clones.front().R_GI);
          const auto& lm = r.landmarks[r.landmarks.size() / 2];
          const Eigen::Vector3d el = T * data.landmarks.at(lm.id).p_G - lm.p_G;
          const int j = 15 + 3 * static_cast<int>(r.landmarks.size() / 2);
          lm_nees.push_back(el.dot(r.P_joint.block<3, 3>(j, j).ldlt().solve(el)));
        }
      }
    }
    ASSERT_GT(nees.size(), 50u);
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    double mean_nees = 0.0;
    for (double x : nees) mean_nees += x / static_cast<double>(nees.size());
    const eval::NeesBand band = eval::neesBand95(8, static_cast<int>(nees.size()));
    std::string why;
    for (const auto& [k, v] : refused) why += " " + k + " " + std::to_string(v);
    std::cout << "[ info ] " << length << " s windows, " << nees.size() << " runs (refused by refinement:"
              << (why.empty() ? " none" : why) << "): gravity error median linear " << median(lin_err) << " deg, refined "
              << median(ref_err) << " deg; NEES (8 dof) " << mean_nees << ", band [" << band.lower << ", "
              << band.upper << "]\n";
    std::sort(nees.begin(), nees.end());
    std::cout << "[ info ]   NEES median " << nees[nees.size() / 2] << ", runs above chi2_0.99(8) = 20.1: "
              << std::count_if(nees.begin(), nees.end(), [](double x) { return x > 20.09; }) << "\n";
    if (length < 0.5) {
      EXPECT_LT(median(ref_err), 0.5 * median(lin_err));  // short windows: refinement is what makes them work
    }
    double mean_lm = 0.0;
    for (double x : lm_nees) mean_lm += x / static_cast<double>(lm_nees.size());
    const eval::NeesBand lm_band = eval::neesBand95(3, static_cast<int>(lm_nees.size()));
    std::cout << "[ info ]   landmark NEES (3 dof, one per run) " << mean_lm << ", band [" << lm_band.lower << ", "
              << lm_band.upper << "]\n";
    if (length >= 1.0) {  // the operating window: accurate and consistent
      EXPECT_GT(mean_lm, lm_band.lower);
      EXPECT_LT(mean_lm, lm_band.upper);
      EXPECT_LT(median(ref_err), 1.0);
      EXPECT_GT(mean_nees, band.lower);
      EXPECT_LT(mean_nees, band.upper);
    }
  }
}

// In place (DynamicInitConfig): a device rotating at up to 0.3 rad/s without
// translating gets its tilt from the gyro-compensated mean specific force;
// the same rotation while translating at 0.3 m/s is refused by the
// rotation-compensated disparity.
TEST(InPlaceInit, RotationWithoutTranslation) {
  const Sophus::SO3d R_GI0 = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 0.7));
  const Sophus::SE3d T_IC;  // camera = IMU frame
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<Eigen::Vector3d> landmarks;
  for (int i = 0; i < 60; ++i) landmarks.push_back(R_GI0 * Eigen::Vector3d(2 * u(rng), 2 * u(rng), 3.5 + 1.5 * u(rng)));
  auto omega = [](double t) { return Eigen::Vector3d(0.3 * std::sin(3 * t), 0.2 * std::cos(2 * t), 0.1); };
  for (const double speed : {0.0, 0.3}) {
    const Eigen::Vector3d v_G(speed, 0.0, 0.0);
    std::vector<ImuSample> imu;
    std::vector<InitFrame> frames;
    Sophus::SO3d R = R_GI0;
    const double dt = 0.005;
    for (int k = 0; k <= 200; ++k) {
      const double t = k * dt;
      imu.push_back({t, omega(t), R.inverse() * (-kGravity)});
      if (k % 10 == 0) {
        InitFrame f;
        f.t = t;
        for (size_t i = 0; i < landmarks.size(); ++i) {
          f.bearings.emplace_back(static_cast<int64_t>(i), (R.inverse() * (landmarks[i] - v_G * t)).normalized());
        }
        frames.push_back(std::move(f));
      }
      R = R * Sophus::SO3d::exp(omega(t + 0.5 * dt) * dt);
    }
    DynamicInitConfig cfg;
    cfg.pixels_per_radian = 500.0;
    double disparity = 0.0;
    const auto st = inPlaceInitialize(imu, frames, T_IC, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), ImuNoise{},
                                      cfg, &disparity);
    if (speed == 0.0) {
      ASSERT_TRUE(st.has_value()) << "disparity " << disparity;
      EXPECT_LT(disparity, 0.1);
      // Tilt: gravity in the IMU frame at the last frame.
      // R ran one step past the last sample; recompute the last attitude.
      Sophus::SO3d R_last = R_GI0;
      for (int k = 0; k < 200; ++k) R_last = R_last * Sophus::SO3d::exp(omega(k * dt + 0.5 * dt) * dt);
      const Eigen::Vector3d g_est = st->nav.R_GI.inverse() * kGravity, g_t = R_last.inverse() * kGravity;
      EXPECT_LT(std::atan2(g_est.cross(g_t).norm(), g_est.dot(g_t)), 1e-3);
      EXPECT_EQ(st->nav.v_GI.norm(), 0.0);
    } else {
      EXPECT_FALSE(st.has_value());
      EXPECT_GT(disparity, 5.0);
    }
  }
}

}  // namespace
}  // namespace msckf
