// Filter feature update (Stage 3): the measurement Jacobians match finite
// differences of the residual, and the left-nullspace projection removes the
// feature.

#include <gtest/gtest.h>

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/filter/feature_update.h"
#include "msckf/filter/observability.h"
#include "msckf/filter/propagator.h"
#include "msckf/geometry/triangulation.h"
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
  AnchoredPoint f;
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
  const Clone& anchor = s.clones().front();
  const Sophus::SE3d T_GA = Sophus::SE3d(anchor.R_GI, anchor.p_GI) * Sophus::SE3d(s.R_IC, s.p_IC);
  const AnchoredPoint f = AnchoredPoint::fromPoint(T_GA, p_f);
  return {std::move(d), std::move(s), tr, clones, f};
}

// Central differences of the residual (r = z - h) in the state and in the
// feature's inverse-depth parameters.
FeatureJacobian finiteDifferences(const Fixture& fx, const AnchoredPoint& f) {
  const auto cam = camera();
  const double eps = 1e-6;
  FeatureJacobian fd;
  fd.H_x.resize(2 * static_cast<long>(fx.clone_of_obs.size()), fx.state.dim());
  fd.H_f.resize(fd.H_x.rows(), 3);
  for (int j = 0; j < fx.state.dim(); ++j) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(fx.state.dim());
    dx(j) = eps;
    FilterState plus = fx.state, minus = fx.state;
    plus.inject(dx);
    minus.inject(-dx);
    FeatureJacobian a, b;
    EXPECT_TRUE(linearizeFeature(plus, *cam, fx.track, fx.clone_of_obs, f, &a));
    EXPECT_TRUE(linearizeFeature(minus, *cam, fx.track, fx.clone_of_obs, f, &b));
    fd.H_x.col(j) = -(a.r - b.r) / (2.0 * eps);
  }
  for (int j = 0; j < 3; ++j) {
    AnchoredPoint plus = f, minus = f;
    plus.x(j) += eps;
    minus.x(j) -= eps;
    FeatureJacobian a, b;
    EXPECT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, plus, &a));
    EXPECT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, minus, &b));
    fd.H_f.col(j) = -(a.r - b.r) / (2.0 * eps);
  }
  return fd;
}

TEST(FilterUpdate, JacobiansMatchFiniteDifferences) {
  Fixture fx = makeFixture(false);
  // Move the estimate off the truth so the residual is not zero.
  fx.state.inject(1e-3 * Eigen::VectorXd::LinSpaced(fx.state.dim(), -1.0, 1.0));
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *camera(), fx.track, fx.clone_of_obs, fx.f, &fj));

  const FeatureJacobian fd = finiteDifferences(fx, fx.f);

  EXPECT_LT((fd.H_x - fj.H_x).norm(), 1e-6 * fj.H_x.norm());
  EXPECT_LT((fd.H_f - fj.H_f).norm(), 1e-6 * fj.H_f.norm());
}

// A point at infinity (rho = 0): the Jacobians stay finite and correct, and
// only clone and extrinsic rotations are constrained.
TEST(FilterUpdate, PointAtInfinityConstrainsRotationOnly) {
  Fixture fx = makeFixture(false);
  AnchoredPoint f = fx.f;
  f.x.z() = 0.0;
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *camera(), fx.track, fx.clone_of_obs, f, &fj));

  const FeatureJacobian fd = finiteDifferences(fx, f);

  EXPECT_LT((fd.H_x - fj.H_x).norm(), 1e-6 * fj.H_x.norm());
  EXPECT_LT((fd.H_f - fj.H_f).norm(), 1e-6 * fj.H_f.norm());
  for (size_t i = 0; i < fx.state.clones().size(); ++i) {
    EXPECT_EQ(fj.H_x.middleCols<3>(fx.state.cloneIndex(i) + 3).norm(), 0.0);
    EXPECT_GT(fj.H_x.middleCols<3>(fx.state.cloneIndex(i)).norm(), 1.0);
  }
  EXPECT_EQ(fj.H_x.middleCols<3>(fx.state.extrinsicIndex() + 3).norm(), 0.0);
}

// Two observations, H_f with a zero rho column (no baseline) and a residual
// orthogonal to H_f: an inconsistent track. Every row of it must survive.
TEST(FilterUpdate, ProjectionKeepsInconsistencyWhenFeatureIsRankDeficient) {
  FeatureJacobian fj;
  fj.H_f = Eigen::MatrixXd::Zero(4, 3);
  fj.H_f.leftCols<2>() << 1.0, 0.2, -0.3, 0.9, 0.8, -0.1, 0.1, 1.1;
  fj.H_x = Eigen::MatrixXd::Identity(4, 6);
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(fj.H_f.leftCols<2>());
  const Eigen::MatrixXd Q = qr.householderQ();
  fj.r = 40.0 * (Q.col(2) + Q.col(3)).normalized();

  const ProjectedResidual p = projectOutFeature(fj);

  EXPECT_EQ(p.r_o.size(), 2);
  EXPECT_NEAR(p.r_o.norm(), 40.0, 1e-9);
}

// A residual that a change of the feature explains is removed entirely.
TEST(FilterUpdate, ProjectionRemovesFeatureMotion) {
  Fixture fx = makeFixture(false);
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *camera(), fx.track, fx.clone_of_obs, fx.f, &fj));
  fj.r = fj.H_f * Eigen::Vector3d(0.01, -0.02, 0.05);

  const ProjectedResidual p = projectOutFeature(fj);

  EXPECT_EQ(p.r_o.size(), 2 * 5 - 3);
  EXPECT_LT(p.r_o.norm(), 1e-12 * fj.H_f.norm());
}

// Each observation's Jacobian annihilates the unobservable directions taken at
// its clone's linearization point (estimates moved off the truth).
TEST(FilterUpdate, MeasurementJacobianAnnihilatesUnobservableDirections) {
  Fixture fx = makeFixture(false);
  fx.state.inject(1e-2 * Eigen::VectorXd::LinSpaced(fx.state.dim(), -1.0, 1.0));
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *camera(), fx.track, fx.clone_of_obs, fx.f, &fj));

  for (size_t j = 0; j < fx.clone_of_obs.size(); ++j) {
    const Clone& c = fx.state.clones()[fx.clone_of_obs[j]];
    NavState x;
    x.R_GI = c.R_GI;
    x.p_GI = c.p_GI;
    const Eigen::MatrixXd N = unobservableDirections(x, {fx.f});
    const Eigen::Matrix<double, 2, 6> H_pose = fj.H_x.block<2, 6>(2 * static_cast<long>(j), fx.state.cloneIndex(j));
    const Eigen::Matrix<double, 2, 3> H_f = fj.H_f.middleRows<2>(2 * static_cast<long>(j));
    const Eigen::MatrixXd HN = H_pose * N.topRows<6>() + H_f * N.bottomRows<3>();
    EXPECT_LT(HN.norm(), 1e-9 * H_pose.norm()) << "observation " << j;
  }
}

// With FEJ, the clone blocks are evaluated at the clones' first estimates:
// an update that moves the clones leaves H unchanged.
TEST(FilterUpdate, FejFreezesCloneJacobians) {
  Fixture fx = makeFixture(true);
  const auto cam = camera();
  FeatureJacobian before, after;
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.f, &before));
  const int c0 = fx.state.cloneIndex(0);
  const int nc = 6 * static_cast<int>(fx.state.clones().size());
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(fx.state.dim());
  dx.segment(c0, nc).setConstant(1e-2);
  fx.state.inject(dx);
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.f, &after));
  EXPECT_LT((before.H_x.middleCols(c0, nc) - after.H_x.middleCols(c0, nc)).norm(), 1e-12);
  EXPECT_GT((before.r - after.r).norm(), 1.0);  // the residual does move
}

// Exact observations at the true state: the update keeps the state there.
TEST(FilterUpdate, ExactObservationsLeaveTruthUnchanged) {
  Fixture fx = makeFixture(true);
  const auto cam = camera();
  const FilterState before = fx.state;
  FeatureJacobian fj;
  ASSERT_TRUE(linearizeFeature(fx.state, *cam, fx.track, fx.clone_of_obs, fx.f, &fj));
  std::cout << "residual at truth " << fj.r.norm() << " px\n";
  EXPECT_LT(fj.r.norm(), 1e-2);  // RK4 propagation error only (~2e-3 px)
  const UpdateStats st = featureUpdate(fx.state, *cam, {fx.track}, UpdateOptions{});
  EXPECT_EQ(st.used, 1) << "triangulation " << st.rejected_triangulation << " chi2 " << st.rejected_chi2;
  EXPECT_EQ(st.rows, 2 * 5 - 3);
  EXPECT_LT((fx.state.nav.p_GI - before.nav.p_GI).norm(), 1e-4);  // P = I: that residual moves it ~2e-5 m
  EXPECT_LT(fx.state.P.trace(), before.P.trace());
}

// One observation has no residual left after the nullspace projection
// (2M - 3 < 0): skipped even when min_observations allows it.
TEST(FilterUpdate, SingleObservationSkipped) {
  Fixture fx = makeFixture(true);
  FeatureTrack one;
  one.id = fx.track.id;
  one.t_cam = {fx.track.t_cam.front()};
  one.uv = {fx.track.uv.front()};
  one.bearing = {fx.track.bearing.front()};
  UpdateOptions options;
  options.min_observations = 1;
  const Eigen::MatrixXd P = fx.state.P;
  const UpdateStats st = featureUpdate(fx.state, *camera(), {one}, options);
  EXPECT_EQ(st.features, 0);
  EXPECT_EQ(fx.state.P, P);
}

TEST(Triangulation, MaxRayAngle) {
  EXPECT_DOUBLE_EQ(maxRayAngleDeg({}), 0.0);
  EXPECT_DOUBLE_EQ(maxRayAngleDeg({Eigen::Vector3d::UnitX()}), 0.0);
  // Lengths do not matter; the widest pair (x, -x + y) is 135 deg.
  EXPECT_NEAR(maxRayAngleDeg({Eigen::Vector3d(2, 0, 0), Eigen::Vector3d(0, 3, 0), Eigen::Vector3d(-1, 1, 0)}), 135.0,
              1e-12);
}

}  // namespace
}  // namespace msckf::filter
