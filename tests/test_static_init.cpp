// Stage 1a: static initialization. Exact on noise-free data, consistent
// (NEES inside the chi-square band) over Monte Carlo runs with noise and bias
// drawn from the prior, and refuses windows that are not stationary.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/core/constants.h"
#include "msckf/eval/nees.h"
#include "msckf/init/static_init.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"
#include "msckf/sim/spline.h"

#include <Eigen/Cholesky>

#include <random>

namespace msckf {
namespace {

constexpr double kWindow = 1.0;

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05,
      0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

// A spline that holds one pose for its whole range.
TrajectoryPtr staticTrajectory(const Sophus::SO3d& R_GI, double duration) {
  constexpr double kKnotDt = 0.1;
  SplineTrajectory spline(0.0, kKnotDt);
  const int n = static_cast<int>(std::ceil(duration / kKnotDt)) + 4;
  for (int k = 0; k < n; ++k) spline.addControlPose(R_GI, Eigen::Vector3d(1.0, 2.0, 1.5));
  return std::make_shared<SplineTrajectory>(spline);
}

SimConfig imuOnlyConfig(double duration, uint64_t seed) {
  SimConfig c;
  c.duration = duration;
  c.cam_rate = 1.0;
  c.ground_density = c.wall_density = c.ceiling_density = 0.0;
  c.seed = seed;
  return c;
}

Sophus::SO3d randomAttitude(std::mt19937_64& rng, double max_tilt) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  return Sophus::SO3d::rotZ(M_PI * u(rng)) * Sophus::SO3d::rotX(max_tilt * u(rng)) *
         Sophus::SO3d::rotY(max_tilt * u(rng));
}

const GroundTruth& truthAt(const SimData& data, double t) {
  for (const GroundTruth& g : data.imu_truth) {
    if (g.t >= t - 1e-9) return g;
  }
  return data.imu_truth.back();
}

// Up vector of the IMU frame: gravity's "up" expressed in I.
Eigen::Vector3d upInImu(const Sophus::SO3d& R_GI) { return R_GI.inverse() * Eigen::Vector3d::UnitZ(); }

TEST(StaticInit, ExactWithoutNoise) {
  std::mt19937_64 rng(5);
  for (int trial = 0; trial < 20; ++trial) {
    const Sophus::SO3d R_true = randomAttitude(rng, 1.2);
    SimConfig sc = imuOnlyConfig(1.5, trial + 1);
    sc.imu_noise_enabled = false;
    sc.gyro_bias0 = Eigen::Vector3d(0.01, -0.02, 0.005);
    sc.acc_bias0 = Eigen::Vector3d(0.05, 0.02, -0.08);
    const SimData data = Simulator(sc, staticTrajectory(R_true, 2.0), camera()).run();

    StaticInitConfig cfg;
    cfg.acc_bias_prior = sc.acc_bias0;
    const auto init = staticInitialize(data.imu, 1.2, cfg);
    ASSERT_TRUE(init) << trial;
    EXPECT_LT((upInImu(init->nav.R_GI) - upInImu(R_true)).norm(), 1e-9);
    EXPECT_LT((init->gyro_bias - sc.gyro_bias0).norm(), 1e-12);
    EXPECT_LT(init->nav.v_GI.norm(), 1e-15);
    // Yaw convention: the IMU x axis projects onto global +x.
    const Eigen::Vector3d x_G = init->nav.R_GI * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(x_G.y(), 0.0, 1e-12);
    EXPECT_GT(x_G.x(), 0.0);
    // Covariance is symmetric positive definite.
    EXPECT_LT((init->P - init->P.transpose()).norm(), 1e-15);
    EXPECT_EQ(Eigen::LLT<Matrix15d>(init->P).info(), Eigen::Success);
  }
}

TEST(StaticInit, ConsistentOverMonteCarlo) {
  constexpr int kRuns = 200;
  std::mt19937_64 rng(17);
  std::normal_distribution<double> normal(0.0, 1.0);
  StaticInitConfig cfg;
  cfg.acc_bias_prior = Eigen::Vector3d(0.02, -0.03, 0.01);
  cfg.acc_bias_sigma = 0.05;
  const double bg_sigma = 0.01;

  double nees_tilt = 0.0, nees_bg = 0.0, nees_all = 0.0;
  for (int run = 0; run < kRuns; ++run) {
    const Sophus::SO3d R_true = randomAttitude(rng, 0.6);
    SimConfig sc = imuOnlyConfig(1.5, 1000 + run);
    sc.gyro_bias0 = bg_sigma * Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    sc.acc_bias0 = cfg.acc_bias_prior +
                   cfg.acc_bias_sigma * Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    cfg.noise = sc.imu_noise;
    const SimData data = Simulator(sc, staticTrajectory(R_true, 2.0), camera()).run();
    const double t_end = 1.2;
    const auto init = staticInitialize(data.imu, t_end, cfg);
    ASSERT_TRUE(init) << run;
    const GroundTruth& gt = truthAt(data, t_end);

    // Tilt error (yaw is gauge): dtheta_tilt = z_true x z_hat to first order,
    // expressed in a basis B orthogonal to z_hat.
    const Eigen::Vector3d z = upInImu(init->nav.R_GI);
    const Eigen::Vector3d dtheta = upInImu(gt.state.R_GI).cross(z);
    Eigen::Matrix<double, 3, 2> B;
    B.col(0) = z.unitOrthogonal();
    B.col(1) = z.cross(B.col(0));

    using namespace imu_err;
    Eigen::Matrix<double, 8, 1> e;
    e << B.transpose() * dtheta, gt.gyro_bias - init->gyro_bias, gt.acc_bias - init->acc_bias;
    Eigen::Matrix<double, 8, 15> J = Eigen::Matrix<double, 8, 15>::Zero();
    J.block<2, 3>(0, kTheta) = B.transpose();
    J.block<3, 3>(2, kBg).setIdentity();
    J.block<3, 3>(5, kBa).setIdentity();
    const Eigen::Matrix<double, 8, 8> C = J * init->P * J.transpose();

    nees_all += e.dot(C.ldlt().solve(e));
    nees_tilt += e.head<2>().dot(C.topLeftCorner<2, 2>().ldlt().solve(e.head<2>()));
    nees_bg += e.segment<3>(2).dot(C.block<3, 3>(2, 2).ldlt().solve(e.segment<3>(2)));
  }
  nees_all /= kRuns;
  nees_tilt /= kRuns;
  nees_bg /= kRuns;
  const auto band8 = eval::neesBand95(8, kRuns), band2 = eval::neesBand95(2, kRuns),
             band3 = eval::neesBand95(3, kRuns);
  std::cout << "[ info ] mean NEES: all (8) " << nees_all << " in [" << band8.lower << ", "
            << band8.upper << "], tilt (2) " << nees_tilt << " in [" << band2.lower << ", "
            << band2.upper << "], gyro bias (3) " << nees_bg << " in [" << band3.lower << ", "
            << band3.upper << "]\n";
  EXPECT_GT(nees_all, band8.lower);
  EXPECT_LT(nees_all, band8.upper);
  EXPECT_GT(nees_tilt, band2.lower);
  EXPECT_LT(nees_tilt, band2.upper);
  EXPECT_GT(nees_bg, band3.lower);
  EXPECT_LT(nees_bg, band3.upper);
}

TEST(StaticInit, RejectsMotionAndDisparity) {
  StaticInitConfig cfg;
  // Handheld motion: the accelerometer test fails.
  {
    const SimData data =
        Simulator(imuOnlyConfig(5.0, 3), makeScenario(Scenario::kHandheld, 6.0, 3), camera()).run();
    StationarityCheck check;
    EXPECT_FALSE(staticInitialize(data.imu, 3.0, cfg, std::nullopt, &check));
    EXPECT_FALSE(check.stationary);
    EXPECT_GT(check.acc_std, cfg.max_acc_std) << check.reason;
  }
  // At rest, but the tracker reports image motion: refused; with low disparity,
  // accepted.
  {
    const SimData data =
        Simulator(imuOnlyConfig(1.5, 4), staticTrajectory(Sophus::SO3d(), 2.0), camera()).run();
    StationarityCheck check;
    EXPECT_FALSE(staticInitialize(data.imu, 1.2, cfg, 3.0, &check));
    EXPECT_NE(check.reason.find("disparity"), std::string::npos) << check.reason;
    EXPECT_TRUE(staticInitialize(data.imu, 1.2, cfg, 0.2));
  }
  // Too few samples.
  {
    std::vector<ImuSample> few(5);
    for (int i = 0; i < 5; ++i) few[i].t = 0.1 * i;
    EXPECT_FALSE(staticInitialize(few, 0.4, cfg));
  }
}

}  // namespace
}  // namespace msckf
