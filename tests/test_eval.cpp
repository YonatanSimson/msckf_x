// Evaluation harness tests: every metric checked on inputs with a known answer.

#include <gtest/gtest.h>

#include "msckf/eval/report.h"
#include "msckf/sim/scenarios.h"

#include <Eigen/Cholesky>

#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

namespace msckf::eval {
namespace {

namespace fs = std::filesystem;

// Ground truth sampled from a simulator trajectory.
Trajectory sampleScenario(Scenario scenario, double duration, double rate) {
  const TrajectoryPtr spline = makeScenario(scenario, duration + 1.0, 9);
  Trajectory out;
  for (int k = 0; k <= static_cast<int>(duration * rate); ++k) {
    const double t = k / rate;
    const TrajectoryState s = spline->evaluate(t);
    out.push_back({t, Sophus::SE3d(s.R_GI, s.p_GI)});
  }
  return out;
}

Association identityAssociation(const Trajectory& gt, const Trajectory& est) {
  Association a;
  for (size_t i = 0; i < gt.size(); ++i) {
    a.t.push_back(gt[i].t);
    a.gt.push_back(gt[i].T_GI);
    a.est.push_back(est[i].T_GI);
    a.est_index.push_back(i);
  }
  return a;
}

class TempDir {
 public:
  TempDir() {
    path_ = fs::temp_directory_path() /
            ("msckf_eval_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(path_);
  }
  ~TempDir() { fs::remove_all(path_); }
  std::string operator/(const std::string& name) const { return (path_ / name).string(); }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

TEST(ChiSquare, MatchesTabulatedQuantiles) {
  struct Case { double p, dof, expected; };
  for (const Case& c : {Case{0.95, 1, 3.841459}, Case{0.95, 2, 5.991465}, Case{0.95, 3, 7.814728},
                        Case{0.975, 3, 9.348404}, Case{0.025, 3, 0.2157953},
                        Case{0.025, 5, 0.8312116}, Case{0.975, 5, 12.83250},
                        Case{0.95, 100, 124.3421}, Case{0.975, 100, 129.5612},
                        Case{0.025, 100, 74.22193}}) {
    EXPECT_NEAR(chiSquareQuantile(c.p, c.dof), c.expected, 1e-5 * c.expected)
        << "p " << c.p << " dof " << c.dof;
  }
  // P(1, x) = 1 - exp(-x), across both the series and continued-fraction branches.
  for (double x : {0.1, 1.0, 1.9, 2.1, 5.0, 30.0}) {
    EXPECT_NEAR(regularizedGammaP(1.0, x), 1.0 - std::exp(-x), 1e-14);
  }
}

TEST(Alignment, RecoversKnownTransforms) {
  const Trajectory gt = sampleScenario(Scenario::kHandheld, 30.0, 20.0);
  const Sophus::SO3d R_yaw = Sophus::SO3d::rotZ(0.7);
  const Sophus::SO3d R_full = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 1.1));
  const Eigen::Vector3d t(1.0, -2.0, 0.5);

  struct Case { AlignmentType type; Sophus::SO3d R; double scale; };
  for (const Case& c : {Case{AlignmentType::kPosYaw, R_yaw, 1.0}, Case{AlignmentType::kSE3, R_full, 1.0},
                        Case{AlignmentType::kSim3, R_full, 1.3}}) {
    // est = inverse of the alignment applied to gt, so align() must return it.
    std::vector<Eigen::Vector3d> est, g;
    for (const StampedPose& s : gt) {
      g.push_back(s.T_GI.translation());
      est.push_back(c.R.inverse() * (s.T_GI.translation() - t) / c.scale);
    }
    const Alignment a = align(est, g, c.type);
    EXPECT_LT((a.R.inverse() * c.R).log().norm(), 1e-10) << toString(c.type);
    EXPECT_LT((a.t - t).norm(), 1e-9) << toString(c.type);
    EXPECT_NEAR(a.scale, c.scale, 1e-10) << toString(c.type);
  }
}

TEST(Ate, NoiseOfKnownSize) {
  const Trajectory gt = sampleScenario(Scenario::kHandheld, 60.0, 20.0);
  std::mt19937 rng(3);
  std::normal_distribution<double> n(0.0, 0.01);
  Trajectory est = gt;
  for (StampedPose& s : est) {
    s.T_GI = Sophus::SE3d(Sophus::SO3d::rotZ(0.5) * s.T_GI.so3(),
                          Sophus::SO3d::rotZ(0.5) * s.T_GI.translation() +
                              Eigen::Vector3d(3.0 + n(rng), n(rng), n(rng)));
  }
  const AteResult ate = computeAte(identityAssociation(gt, est), AlignmentType::kPosYaw);
  // Per-axis sigma 1 cm -> RMSE of the 3D error ~ sqrt(3) cm after removing the gauge.
  EXPECT_NEAR(ate.position.rmse, 0.01 * std::sqrt(3.0), 0.001);
  // The yaw fitted to noisy positions is slightly off, which shows up in rotation.
  EXPECT_LT(ate.rotation.rmse, 0.05);
}

TEST(RelativeError, ScaleAndRotationDrift) {
  // Straight line at 1 m/s along x for 100 s.
  Trajectory gt;
  for (int k = 0; k <= 2000; ++k) {
    const double t = k * 0.05;
    gt.push_back({t, Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(t, 0.0, 1.0))});
  }
  // 1 % scale error: every segment has 1 % translation error and no rotation error.
  Trajectory scaled = gt;
  for (StampedPose& s : scaled) s.T_GI.translation() *= 1.01;
  for (const auto& r : computeRelativeError(identityAssociation(gt, scaled), {10.0, 25.0, 40.0})) {
    EXPECT_NEAR(r.translation_pct.mean, 1.0, 0.03) << r.length;
    EXPECT_NEAR(r.translation_pct.max, 1.0, 0.03) << r.length;
    EXPECT_LT(r.rotation_deg_per_m.max, 1e-9);
    EXPECT_GT(r.translation_pct.count, 100u);
  }
  // Heading drift of 0.2 deg per meter traveled: rotation error is 0.2 deg/m.
  Trajectory drift = gt;
  for (StampedPose& s : drift) {
    s.T_GI.so3() = Sophus::SO3d::rotZ(0.2 * M_PI / 180.0 * s.T_GI.translation().x());
  }
  for (const auto& r : computeRelativeError(identityAssociation(gt, drift), {10.0, 25.0})) {
    EXPECT_NEAR(r.rotation_deg_per_m.mean, 0.2, 0.002) << r.length;
  }
  EXPECT_EQ(defaultSegmentLengths(100.0), (std::vector<double>{10, 20, 30, 40, 50}));
  EXPECT_EQ(defaultSegmentLengths(29.99), (std::vector<double>{3, 6, 9, 12, 15}));
  EXPECT_EQ(defaultSegmentLengths(7.0), (std::vector<double>{0.5, 1.5, 2.0, 3.0, 3.5}));
}

TEST(Trajectory, InterpolationAndAssociation) {
  const TrajectoryPtr spline = makeScenario(Scenario::kHandheld, 11.0, 9);
  Trajectory gt;
  for (int k = 0; k <= 2000; ++k) {  // 200 Hz ground truth
    const TrajectoryState s = spline->evaluate(k / 200.0);
    gt.push_back({k / 200.0, Sophus::SE3d(s.R_GI, s.p_GI)});
  }
  double max_p = 0.0, max_r = 0.0;
  Trajectory est;
  for (double t = 0.0013; t < 10.0; t += 0.0497) {  // off-grid estimate times
    Sophus::SE3d T;
    ASSERT_TRUE(interpolatePose(gt, t, 0.01, T));
    const TrajectoryState s = spline->evaluate(t);
    max_p = std::max(max_p, (T.translation() - s.p_GI).norm());
    max_r = std::max(max_r, (s.R_GI.inverse() * T.so3()).log().norm());
    est.push_back({t, T});
  }
  // Interpolation error at 200 Hz is O(a dt^2 / 8): well below a millimeter.
  EXPECT_LT(max_p, 1e-4);
  EXPECT_LT(max_r, 1e-4);

  Sophus::SE3d T;
  EXPECT_FALSE(interpolatePose(gt, -1.0, 0.01, T));
  EXPECT_FALSE(interpolatePose(gt, 0.0025, 0.001, T));  // gap 5 ms > 1 ms
  est.push_back({50.0, Sophus::SE3d()});  // outside ground truth: dropped
  EXPECT_EQ(associate(est, gt, 0.01).size(), est.size() - 1);
}

TEST(Trajectory, SameGridTimestampsMatchDespiteRounding) {
  // Ground truth and estimate on the same 20 Hz grid, but ground-truth stamps
  // went through integer nanoseconds: max_gap equal to the spacing must still
  // match every pose.
  Trajectory gt, est;
  for (int k = 0; k <= 1200; ++k) {
    const double t = k * 0.05;
    const Sophus::SE3d T(Sophus::SO3d::rotZ(0.01 * k), Eigen::Vector3d(t, 0.0, 0.0));
    gt.push_back({static_cast<double>(std::llround(t * 1e9)) * 1e-9 + 3e-10, T});
    est.push_back({t, T});
  }
  const Association a = associate(est, gt, 0.05);
  EXPECT_EQ(a.size(), est.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_LT((a.gt[i].inverse() * a.est[i]).log().norm(), 1e-12);
  }
}

TEST(Trajectory, FileRoundTrips) {
  TempDir dir;
  const Trajectory gt = sampleScenario(Scenario::kCart, 20.0, 10.0);
  writeTum(dir / "traj.txt", gt);
  const Trajectory back = readTum(dir / "traj.txt");
  ASSERT_EQ(back.size(), gt.size());
  for (size_t i = 0; i < gt.size(); ++i) {
    EXPECT_DOUBLE_EQ(back[i].t, gt[i].t);
    EXPECT_LT((back[i].T_GI.translation() - gt[i].T_GI.translation()).norm(), 1e-12);
    EXPECT_LT((back[i].T_GI.so3().inverse() * gt[i].T_GI.so3()).log().norm(), 1e-12);
  }

  std::ofstream euroc(dir / "data.csv");
  euroc << "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], "
           "q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1]\n"
        << "1403636579758555392,4.688319,-1.786938,0.783338,0.534108,-0.153029,-0.827383,"
           "-0.082152,-0.027876\n";
  euroc.close();
  const Trajectory e = readEurocGroundTruth(dir / "data.csv");
  ASSERT_EQ(e.size(), 1u);
  EXPECT_NEAR(e[0].t, 1403636579.758555392, 1e-6);
  EXPECT_NEAR(e[0].T_GI.translation().x(), 4.688319, 1e-12);
  EXPECT_NEAR(e[0].T_GI.unit_quaternion().w(), 0.534108, 1e-5);  // after normalization
}

TEST(Nees, MonteCarloConsistentAndOverconfident) {
  constexpr int kRuns = 50, kEpochs = 200;
  std::mt19937 rng(17);
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.5, 2.0);

  MonteCarloNees consistent, overconfident;
  for (int run = 0; run < kRuns; ++run) {
    std::vector<PoseNees> ok, over;
    for (int k = 0; k < kEpochs; ++k) {
      // Random SPD covariance with correlated blocks.
      Matrix6d A;
      for (int i = 0; i < 36; ++i) A(i) = 0.05 * n(rng);
      Matrix6d P = A * A.transpose();
      for (int i = 0; i < 6; ++i) P(i, i) += 1e-3 * u(rng);
      Eigen::Matrix<double, 6, 1> z;
      for (int i = 0; i < 6; ++i) z[i] = n(rng);
      const Eigen::Matrix<double, 6, 1> e = P.llt().matrixL() * z;

      const Sophus::SE3d T_est(Sophus::SO3d::exp(Eigen::Vector3d(n(rng), n(rng), n(rng))),
                               Eigen::Vector3d(n(rng), n(rng), n(rng)));
      // Right perturbation: R_true = R_est Exp(dtheta), p_true = p_est + dp.
      const Sophus::SE3d T_true(T_est.so3() * Sophus::SO3d::exp(e.head<3>()),
                                T_est.translation() + e.tail<3>());
      ok.push_back(poseNees(T_est, P, T_true));
      over.push_back(poseNees(T_est, P / 4.0, T_true));
    }
    consistent.addRun(ok);
    overconfident.addRun(over);
  }
  const auto s = consistent.summarize();
  EXPECT_NEAR(s.mean_pose, 6.0, 0.15);
  EXPECT_NEAR(s.mean_orientation, 3.0, 0.1);
  EXPECT_NEAR(s.mean_position, 3.0, 0.1);
  EXPECT_GT(s.inside_pose, 0.88);  // nominally 95 %
  EXPECT_NEAR(s.band6.lower, chiSquareQuantile(0.025, 300) / 50, 1e-9);

  const auto o = overconfident.summarize();
  EXPECT_NEAR(o.mean_pose, 24.0, 0.6);
  EXPECT_LT(o.inside_pose, 0.01);
}

TEST(Timing, PercentilesAndScopedTimer) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  EXPECT_DOUBLE_EQ(percentile(v, 0.5), 50.5);
  EXPECT_NEAR(percentile(v, 0.99), 99.01, 1e-12);

  LatencyRecorder rec;
  {
    ScopedTimer timer(rec, "sleep");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  rec.add("update", 0.002);
  rec.add("update", 0.004);
  const auto s = rec.summarize();
  EXPECT_EQ(s.at("sleep").count, 1u);
  EXPECT_GE(s.at("sleep").median, 0.005);
  EXPECT_DOUBLE_EQ(s.at("update").median, 0.003);

  CpuLoadMeter cpu;
  volatile double x = 0.0;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
  while (std::chrono::steady_clock::now() < end) x = x + 1.0;  // busy one core
  EXPECT_GT(cpu.load(), 0.5);
}

TEST(Report, DatasetListToFiles) {
  TempDir dir;
  const Trajectory gt = sampleScenario(Scenario::kCart, 60.0, 20.0);
  Trajectory est = gt;
  for (StampedPose& s : est) s.T_GI.translation() *= 1.02;
  writeTum(dir / "gt.txt", gt);
  writeTum(dir / "est.txt", est);
  {
    std::ofstream list(dir / "datasets.txt");
    list << "# name gt est [format]\n"
         << "cart_sim gt.txt est.txt tum\n"
         << "missing gt.txt nope.txt\n";
  }
  const std::vector<DatasetEntry> entries = readDatasetList(dir / "datasets.txt");
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].gt_path, dir / "gt.txt");

  Report report;
  for (const auto& e : entries) report.datasets.push_back(evaluateDataset(e, report.options));
  EXPECT_TRUE(report.datasets[0].error.empty());
  EXPECT_FALSE(report.datasets[1].error.empty());
  EXPECT_EQ(report.datasets[0].num_matched, gt.size());
  // A 2 % scale error moves a segment's end by 2 % of its displacement, which is
  // the full path length only on straight segments; U-turns make it shorter.
  for (const auto& r : report.datasets[0].relative) {
    EXPECT_LE(r.translation_pct.max, 2.0 + 1e-9);
    EXPECT_NEAR(r.translation_pct.max, 2.0, 0.02);
    EXPECT_GT(r.translation_pct.mean, 1.0);
  }

  writeReport(dir / "out", report);
  std::ifstream csv(dir / "out/metrics.csv");
  std::string header, row1, row2;
  std::getline(csv, header);
  std::getline(csv, row1);
  std::getline(csv, row2);
  EXPECT_EQ(header.rfind("dataset,error,matched", 0), 0u);
  EXPECT_EQ(row1.rfind("cart_sim,,", 0), 0u);
  EXPECT_EQ(row2.rfind("missing,", 0), 0u);
  EXPECT_TRUE(fs::exists(dir / "out/report.md"));
}

}  // namespace
}  // namespace msckf::eval
