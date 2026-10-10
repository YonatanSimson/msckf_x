// Stage 0 camera-model gates: unproject -> project round trip over every pixel
// (< 1e-6 px) and analytic Jacobians vs central differences on 1e5 random valid
// points (relative error < 1e-6).

#include <gtest/gtest.h>

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/eucm.h"
#include "msckf/camera/kannala_brandt.h"
#include "msckf/camera/pinhole_radtan.h"

#include <functional>
#include <random>

namespace msckf {
namespace {

struct CameraCase {
  std::string label;
  std::function<CameraModelPtr()> make;
  bool expect_beyond_90deg;  // the image contains rays with z < 0
};

Eigen::VectorXd vec(std::initializer_list<double> v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  Eigen::Index i = 0;
  for (double x : v) out[i++] = x;
  return out;
}

// EuRoC cam0 and TUM-VI cam0 calibrations as published (values quoted from
// memory; the gates here only need realistic parameters). The EUCM and the
// alpha < 0.5 cases are synthetic, to exercise the other validity branch.
std::vector<CameraCase> cameraCases() {
  return {
      {"EurocRadtan",
       [] {
         return std::make_shared<PinholeRadtan>(
             752, 480,
             vec({458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359,
                  1.76187114e-05, 0.0}));
       },
       false},
      {"RadtanWithK3",
       [] {
         return std::make_shared<PinholeRadtan>(
             640, 480, vec({400.0, 401.0, 321.0, 239.0, -0.3, 0.1, 1e-3, -5e-4, -0.02}));
       },
       false},
      {"TumviKb4",
       [] {
         return std::make_shared<KannalaBrandt>(
             512, 512,
             vec({190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504,
                  0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,
                  0.00020293673591811182}));
       },
       true},
      {"TumviDoubleSphere",
       [] {
         return std::make_shared<DoubleSphere>(
             512, 512,
             vec({158.28600034966977, 158.2743455478755, 254.96116578191653, 256.8894218032305,
                  -0.17213086034353243, 0.5931177593944744}));
       },
       true},
      {"DoubleSphereAlphaBelowHalf",
       [] {
         return std::make_shared<DoubleSphere>(640, 480,
                                               vec({250.0, 250.0, 320.0, 240.0, 0.3, 0.4}));
       },
       false},
      {"Eucm",
       [] {
         return std::make_shared<Eucm>(512, 512, vec({157.9, 157.8, 254.9, 256.9, 0.58, 1.08}));
       },
       true},
      {"EucmAlphaBelowHalf",
       [] { return std::make_shared<Eucm>(640, 480, vec({300.0, 300.0, 320.0, 240.0, 0.4, 1.2})); },
       false},
  };
}

class CameraModelTest : public ::testing::TestWithParam<CameraCase> {
 protected:
  void SetUp() override { cam_ = GetParam().make(); }

  // Uniform direction times random depth, kept only if it projects into the image
  // and the finite-difference stencil (radius h) stays inside the valid region.
  Eigen::Vector3d randomVisiblePoint(double h_rel) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> depth(0.2, 20.0);
    while (true) {
      const Eigen::Vector3d p = Eigen::Vector3d(n(rng_), n(rng_), n(rng_)).normalized() * depth(rng_);
      if (!cam_->visible(p)) continue;
      const double h = h_rel * p.norm();
      bool stencil_ok = true;
      for (int k = 0; k < 3 && stencil_ok; ++k) {
        stencil_ok = cam_->validPoint(p + h * Eigen::Vector3d::Unit(k)) &&
                     cam_->validPoint(p - h * Eigen::Vector3d::Unit(k));
      }
      if (stencil_ok) return p;
    }
  }

  CameraModelPtr cam_;
  std::mt19937 rng_{7};
};

TEST_P(CameraModelTest, UnprojectProjectRoundTripEveryPixel) {
  double max_err = 0.0;
  int valid = 0, beyond_90 = 0;
  for (int v = 0; v < cam_->height(); ++v) {
    for (int u = 0; u < cam_->width(); ++u) {
      const Eigen::Vector2d uv(u, v);
      Eigen::Vector3d bearing;
      if (!cam_->unproject(uv, bearing)) continue;
      ++valid;
      ASSERT_NEAR(bearing.norm(), 1.0, 1e-12);
      if (bearing.z() < 0.0) ++beyond_90;
      Eigen::Vector2d uv2;
      ASSERT_TRUE(cam_->project(bearing, uv2)) << "pixel " << uv.transpose();
      max_err = std::max(max_err, (uv2 - uv).norm());
    }
  }
  const int total = cam_->width() * cam_->height();
  RecordProperty("valid_pixels", valid);
  RecordProperty("beyond_90deg_pixels", beyond_90);
  std::cout << "[ info ] valid " << valid << "/" << total << ", beyond 90 deg " << beyond_90
            << ", max round-trip error " << max_err << " px\n";
  EXPECT_LT(max_err, 1e-6);
  EXPECT_GT(valid, total / 2);
  if (GetParam().expect_beyond_90deg) {
    EXPECT_GT(beyond_90, 0);
  } else {
    EXPECT_EQ(beyond_90, 0);
  }
}

TEST_P(CameraModelTest, ProjectUnprojectRoundTripRandomPoints) {
  for (int i = 0; i < 10000; ++i) {
    const Eigen::Vector3d p = randomVisiblePoint(0.0);
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam_->project(p, uv));
    Eigen::Vector3d bearing;
    ASSERT_TRUE(cam_->unproject(uv, bearing)) << "uv " << uv.transpose();
    EXPECT_LT((bearing - p.normalized()).norm(), 1e-9);
  }
}

TEST_P(CameraModelTest, PointJacobianMatchesFiniteDifferences) {
  constexpr double kHRel = 1e-5;
  double max_rel = 0.0;
  for (int i = 0; i < 100000; ++i) {
    const Eigen::Vector3d p = randomVisiblePoint(kHRel);
    Eigen::Vector2d uv;
    Matrix23d J;
    ASSERT_TRUE(cam_->project(p, uv, &J));

    const double h = kHRel * p.norm();
    Matrix23d J_fd;
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector2d up, um;
      ASSERT_TRUE(cam_->project(p + h * Eigen::Vector3d::Unit(k), up));
      ASSERT_TRUE(cam_->project(p - h * Eigen::Vector3d::Unit(k), um));
      J_fd.col(k) = (up - um) / (2.0 * h);
    }
    max_rel = std::max(max_rel, (J - J_fd).norm() / J.norm());
  }
  std::cout << "[ info ] max relative point-Jacobian error " << max_rel << "\n";
  EXPECT_LT(max_rel, 1e-6);
}

TEST_P(CameraModelTest, ParamJacobianMatchesFiniteDifferences) {
  constexpr int kPoints = 100000;
  const Eigen::VectorXd params0 = cam_->params();
  const int n = cam_->numParams();

  // Margin so small parameter changes can't push a point out of validity.
  std::vector<Eigen::Vector3d> points(kPoints);
  std::vector<Matrix2Xd> J(kPoints), J_fd(kPoints, Matrix2Xd(2, n));
  for (int i = 0; i < kPoints; ++i) {
    points[i] = randomVisiblePoint(1e-3);
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam_->project(points[i], uv, nullptr, &J[i]));
    ASSERT_EQ(J[i].cols(), n);
  }

  // One parameter perturbation at a time over all points, since setParams may
  // recompute derived quantities such as the validity radius.
  for (int k = 0; k < n; ++k) {
    // Step floor of 1e-5: distortion coefficients sit near zero, and a smaller
    // absolute step lets pixel round-off (~1e-13) dominate the difference.
    const double h = 1e-5 * std::max(std::abs(params0[k]), 1.0);
    Eigen::VectorXd pp = params0, pm = params0;
    pp[k] += h;
    pm[k] -= h;
    std::vector<Eigen::Vector2d> up(kPoints), um(kPoints);
    std::vector<char> ok(kPoints);
    cam_->setParams(pp);
    for (int i = 0; i < kPoints; ++i) ok[i] = cam_->project(points[i], up[i]);
    cam_->setParams(pm);
    for (int i = 0; i < kPoints; ++i) ok[i] = ok[i] && cam_->project(points[i], um[i]);
    cam_->setParams(params0);
    for (int i = 0; i < kPoints; ++i) {
      ASSERT_TRUE(ok[i]) << "param " << k << " point " << points[i].transpose();
      J_fd[i].col(k) = (up[i] - um[i]) / (2.0 * h);
    }
  }

  double max_rel = 0.0;
  for (int i = 0; i < kPoints; ++i) {
    max_rel = std::max(max_rel, (J[i] - J_fd[i]).norm() / J[i].norm());
  }
  std::cout << "[ info ] max relative param-Jacobian error " << max_rel << "\n";
  EXPECT_LT(max_rel, 1e-6);
}

TEST_P(CameraModelTest, RejectsPointsOutsideValidRegion) {
  Eigen::Vector2d uv;
  EXPECT_FALSE(cam_->project(Eigen::Vector3d::Zero(), uv));
  EXPECT_FALSE(cam_->project(Eigen::Vector3d(0.0, 0.0, -1.0), uv));
  EXPECT_FALSE(cam_->validPixel(Eigen::Vector2d(-1.0, 0.0)));
  EXPECT_FALSE(cam_->validPixel(Eigen::Vector2d(0.0, cam_->height())));
}

INSTANTIATE_TEST_SUITE_P(AllModels, CameraModelTest, ::testing::ValuesIn(cameraCases()),
                         [](const auto& info) { return info.param.label; });

}  // namespace
}  // namespace msckf
