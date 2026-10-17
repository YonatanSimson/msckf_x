// Triangulation with anchored inverse depth (plan, Stage 3 "Feature update").

#include <gtest/gtest.h>

#include "msckf/geometry/triangulation.h"

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
  const Eigen::Vector3d X(4.0 * std::sin(2.0944), 0.5, 4.0 * std::cos(2.0944));
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

TEST(Triangulation, RejectsPointBehindCameras) {
  const Rays rays = raysTo({0.4, -0.3, 5.0}, {{0, 0, 0}, {0.3, 0, 0}}, Sophus::SO3d());
  const std::vector<Eigen::Vector3d> flipped = {-rays.bearings[0], -rays.bearings[1]};
  Eigen::Vector3d est;
  double parallax = 0.0;

  EXPECT_FALSE(triangulate(rays.T_GC, flipped, &est, &parallax));
}

}  // namespace
}  // namespace msckf
