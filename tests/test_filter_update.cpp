// Filter feature update (Stage 3): the measurement Jacobians match finite
// differences of the residual, and the left-nullspace projection removes the
// feature.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/filter/feature_update.h"
#include "msckf/filter/propagator.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <Eigen/Dense>

namespace msckf::filter {
namespace {

CameraModelPtr camera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

struct Fixture {
  SimData data;
  FilterState state;
  FeatureTrack track;
  std::vector<size_t> clone_of_obs;
  Eigen::Vector3d p_f;
};

// Noise-free handheld run; clones at five frames 0.2 s apart and a landmark
// seen in all.
Fixture makeFixture(bool fej) {
  SimConfig c = defaultSimConfig(Scenario::kHandheld);
  c.duration = 4.0;
  c.imu_noise_enabled = false;
  c.pixel_noise = 0.0;
  const auto cam = camera();
  const Simulator sim(c, makeScenario(Scenario::kHandheld, 5.0, 7), cam);
  SimData d = sim.run();
  ImuInitState init;
  init.t = d.frames[20].t_cam;
  init.nav = sim.trueState(init.t);
  init.P.setIdentity();
  StateOptions options;
  options.fej = fej;
  FilterState s(init, c.R_IC, c.p_IC, c.time_offset, options, Eigen::Matrix<double, 7, 7>::Identity());
  std::map<int64_t, int> seen;
  for (int f = 20; f < 40; f += 4) {
    const Eigen::Vector3d omega = propagate(s, d.imu, d.frames[f].t_cam, ImuNoise{});
    s.addClone(d.frames[f].t_cam, omega);
    for (const auto& o : d.frames[f].observations) ++seen[o.landmark_id];
  }
  int64_t id = -1;
  for (const auto& [lid, n] : seen) {
    if (n == 5) id = lid;
  }
  EXPECT_GE(id, 0);
  FeatureTrack tr;
  tr.id = id;
  std::vector<size_t> clones;
  for (int f = 20; f < 40; f += 4) {
    for (const auto& o : d.frames[f].observations) {
      if (o.landmark_id != id) continue;
      tr.t_cam.push_back(d.frames[f].t_cam);
      tr.uv.push_back(o.uv);
      Eigen::Vector3d b;
      cam->unproject(o.uv, b);
      tr.bearing.push_back(b);
      clones.push_back(static_cast<size_t>((f - 20) / 4));
    }
  }
  Eigen::Vector3d p_f = Eigen::Vector3d::Zero();
  for (const auto& l : d.landmarks) {
    if (l.id == id) p_f = l.p_G;
  }
  return {std::move(d), std::move(s), tr, clones, p_f};
}

TEST(FilterUpdate, JacobiansMatchFiniteDifferences) {
  Fixture fx = makeFixture(false);
  // Move the estimate off the truth so the residual is not zero.
  Eigen::VectorXd dx0 = 1e-3 * Eigen::VectorXd::LinSpaced(fx.state.dim(), -1.0, 1.0);
  fx.state.inject(dx0);
  const auto cam = camera();
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f, &fj));

  const double eps = 1e-6;
  Eigen::MatrixXd fd_x(fj.r.size(), fx.state.dim());
  for (int j = 0; j < fx.state.dim(); ++j) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(fx.state.dim());
    dx(j) = eps;
    FilterState plus = fx.state, minus = fx.state;
    plus.inject(dx);
    minus.inject(-dx);
    FeatureJacobian a, b;
    ASSERT_TRUE(linearizeFeature(plus, *cam, fx.track, fx.clone_of_obs, fx.p_f, &a));
    ASSERT_TRUE(linearizeFeature(minus, *cam, fx.track, fx.clone_of_obs, fx.p_f, &b));
    fd_x.col(j) = -(a.r - b.r) / (2.0 * eps);  // r = z - h(x)
  }
  Eigen::MatrixXd fd_f(fj.r.size(), 3);
  for (int j = 0; j < 3; ++j) {
    FeatureJacobian a, b;
    const Eigen::Vector3d d = eps * Eigen::Vector3d::Unit(j);
    ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f + d, &a));
    ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f - d, &b));
    fd_f.col(j) = -(a.r - b.r) / (2.0 * eps);
  }
  EXPECT_LT((fd_x - fj.H_x).norm(), 1e-6 * fj.H_x.norm());
  EXPECT_LT((fd_f - fj.H_f).norm(), 1e-6 * fj.H_f.norm());
}

// With FEJ, the clone blocks are evaluated at the clones' first estimates:
// an update that moves the clones leaves H unchanged.
TEST(FilterUpdate, FejFreezesCloneJacobians) {
  Fixture fx = makeFixture(true);
  const auto cam = camera();
  FeatureJacobian before, after;
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f, &before));
  const int c0 = fx.state.cloneIndex(0);
  const int nc = 6 * static_cast<int>(fx.state.clones().size());
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(fx.state.dim());
  dx.segment(c0, nc).setConstant(1e-2);
  fx.state.inject(dx);
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f, &after));
  EXPECT_LT((before.H_x.middleCols(c0, nc) - after.H_x.middleCols(c0, nc)).norm(), 1e-12);
  EXPECT_GT((before.r - after.r).norm(), 1.0);  // the residual does move
}

// Exact observations at the true state: the update keeps the state there.
TEST(FilterUpdate, ExactObservationsLeaveTruthUnchanged) {
  Fixture fx = makeFixture(true);
  const auto cam = camera();
  const FilterState before = fx.state;
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.p_f, &fj));
  std::cout << "residual at truth " << fj.r.norm() << " px\n";
  EXPECT_LT(fj.r.norm(), 1e-2);  // RK4 propagation error only (~2e-3 px)
  const UpdateStats st = featureUpdate(fx.state, *cam, {fx.track}, UpdateOptions{});
  EXPECT_EQ(st.used, 1) << "triangulation " << st.rejected_triangulation << " chi2 " << st.rejected_chi2;
  EXPECT_EQ(st.rows, 2 * 5 - 3);
  EXPECT_LT((fx.state.nav.p_GI - before.nav.p_GI).norm(), 1e-4);  // P = I: that residual moves it ~2e-5 m
  EXPECT_LT(fx.state.P.trace(), before.P.trace());
}

}  // namespace
}  // namespace msckf::filter
