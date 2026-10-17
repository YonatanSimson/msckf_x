// Triangulation with anchored inverse depth (plan, Stage 3 "Feature update").

#include <gtest/gtest.h>

#include "msckf/geometry/triangulation.h"

#include <cmath>
#include <vector>

namespace msckf {
namespace {

struct Rays {
  std::vector<Sophus::SE3d> T_GC;
  std::vector<Eigen::Vector3d> bearings;
};

// Exact bearings of X from cameras at the given centres, all with rotation R_GC.
Rays raysTo(const Eigen::Vector3d& X, const std::vector<Eigen::Vector3d>& centres, const Sophus::SO3d& R_GC) {
  Rays rays;
  for (const Eigen::Vector3d& c : centres) {
    rays.T_GC.emplace_back(R_GC, c);
    rays.bearings.push_back((rays.T_GC.back().inverse() * X).normalized());
  }
  return rays;
}

TEST(Triangulation, RecoversPointFromExactBearings) {
  const Eigen::Vector3d X(0.4, -0.3, 5.0);
  const Rays rays = raysTo(X, {{0, 0, 0}, {0.3, 0, 0}, {0.6, 0.1, 0}}, Sophus::SO3d());
  Eigen::Vector3d est;
  double parallax = 0.0;

  ASSERT_TRUE(triangulate(rays.T_GC, rays.bearings, &est, &parallax));

  EXPECT_LT((est - X).norm(), 1e-9);
}

TEST(Triangulation, RecoversPointPastNinetyDegreesOffAxis) {
  // 120 deg off the anchor's optical axis: no point on its z = 1 plane.
  const double off_axis = 2.0 * M_PI / 3.0;
  const Eigen::Vector3d X(4.0 * std::sin(off_axis), 0.5, 4.0 * std::cos(off_axis));
  const Rays rays = raysTo(X, {{0, 0, 0}, {0, 0.3, 0}, {0, 0.6, 0.2}}, Sophus::SO3d());
  Eigen::Vector3d est;
  double parallax = 0.0;

  ASSERT_TRUE(triangulate(rays.T_GC, rays.bearings, &est, &parallax));

  EXPECT_LT((est - X).norm(), 1e-9);
}

TEST(Triangulation, RecoversFarPointAtLowParallax) {
  const Eigen::Vector3d X(1.0, 2.0, 500.0);
  const Rays rays = raysTo(X, {{0, 0, 0}, {0.05, 0, 0}, {0.1, 0, 0}}, Sophus::SO3d::rotX(0.2));
  Eigen::Vector3d est;
  double parallax = 0.0;

  ASSERT_TRUE(triangulate(rays.T_GC, rays.bearings, &est, &parallax));

  EXPECT_LT(parallax, 0.02);
  EXPECT_LT((est - X).norm() / X.norm(), 1e-6);
}

TEST(Triangulation, RejectsParallelRays) {
  // Cameras at one centre looking at one point: identical rays in the world.
  const Eigen::Vector3d X(0.4, -0.3, 5.0);
  Rays rays;
  for (const Sophus::SO3d& R : {Sophus::SO3d(), Sophus::SO3d::rotY(0.1), Sophus::SO3d::rotX(0.1)}) {
    rays.T_GC.emplace_back(R, Eigen::Vector3d::Zero());
    rays.bearings.push_back((R.inverse() * X).normalized());
  }
  Eigen::Vector3d est;
  double parallax = 0.0;

  EXPECT_FALSE(triangulate(rays.T_GC, rays.bearings, &est, &parallax));
}

TEST(Triangulation, RejectsPointBehindCameras) {
  const Rays rays = raysTo({0.4, -0.3, 5.0}, {{0, 0, 0}, {0.3, 0, 0}}, Sophus::SO3d());
  const std::vector<Eigen::Vector3d> flipped = {-rays.bearings[0], -rays.bearings[1]};
  Eigen::Vector3d est;
  double parallax = 0.0;

  EXPECT_FALSE(triangulate(rays.T_GC, flipped, &est, &parallax));
}

}  // namespace
}  // namespace msckf
