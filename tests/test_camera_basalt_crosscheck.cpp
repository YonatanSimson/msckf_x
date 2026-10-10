// Stage 0 gate: our camera models agree with Basalt's (BSD-3) on the same
// parameters and points to < 1e-9 px. Basalt is used here as an independent
// reference only; nothing from it is linked into the library.
//
// Validity regions are defined independently in each library (ours: the model
// papers plus our own radtan fold analysis), so points where only one side says
// "valid" are counted and reported, not compared.

#include <gtest/gtest.h>

#include "camera_test_cases.h"

#include <basalt/camera/double_sphere_camera.hpp>
#include <basalt/camera/extended_camera.hpp>
#include <basalt/camera/kannala_brandt_camera4.hpp>
#include <basalt/camera/pinhole_radtan8_camera.hpp>

#include <random>

namespace msckf {
namespace {

using test::CameraCase;
using test::cameraCases;

// Type-erased Basalt camera with our parameter layout.
struct BasaltReference {
  std::function<bool(const Eigen::Vector3d&, Eigen::Vector2d&, Matrix23d*, Matrix2Xd*)> project;
  std::function<bool(const Eigen::Vector2d&, Eigen::Vector3d&)> unproject;
};

template <typename BasaltCam>
BasaltReference wrap(const BasaltCam& cam, int our_num_params) {
  BasaltReference ref;
  ref.project = [cam, our_num_params](const Eigen::Vector3d& p, Eigen::Vector2d& uv, Matrix23d* J_point,
                                      Matrix2Xd* J_params) {
    Matrix23d Jp;
    typename BasaltCam::Mat2N Jk;
    const bool ok = cam.project(p, uv, &Jp, &Jk);
    if (J_point) *J_point = Jp;
    // Basalt's radtan8 has k4..k6 after our nine parameters; drop them.
    if (J_params) *J_params = Jk.leftCols(our_num_params);
    return ok;
  };
  ref.unproject = [cam](const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) {
    Eigen::Vector3d p;
    if (!cam.unproject(uv, p)) return false;
    bearing = p.normalized();
    return true;
  };
  return ref;
}

BasaltReference makeReference(const CameraModel& cam) {
  const Eigen::VectorXd k = cam.params();
  const int n = cam.numParams();
  if (cam.name() == "pinhole-radtan") {
    Eigen::Matrix<double, 12, 1> p = Eigen::Matrix<double, 12, 1>::Zero();
    p.head<9>() = k;  // fx fy cx cy k1 k2 p1 p2 k3, then k4 k5 k6 = 0
    return wrap(basalt::PinholeRadtan8Camera<double>(p), n);
  }
  if (cam.name() == "kannala-brandt") {
    return wrap(basalt::KannalaBrandtCamera4<double>(k), n);
  }
  if (cam.name() == "double-sphere") {
    return wrap(basalt::DoubleSphereCamera<double>(k), n);
  }
  if (cam.name() == "eucm") {
    return wrap(basalt::ExtendedUnifiedCamera<double>(k), n);
  }
  throw std::invalid_argument("no Basalt reference for " + cam.name());
}

class BasaltCrossCheckTest : public ::testing::TestWithParam<CameraCase> {
 protected:
  void SetUp() override {
    cam_ = GetParam().make();
    ref_ = makeReference(*cam_);
  }
  CameraModelPtr cam_;
  BasaltReference ref_;
};

TEST_P(BasaltCrossCheckTest, ProjectionAndJacobiansAgree) {
  constexpr int kPoints = 200000;
  std::mt19937 rng(11);
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> depth(0.2, 20.0);

  int compared = 0, only_ours = 0, only_basalt = 0;
  double max_px = 0.0, max_J_point = 0.0, max_J_params = 0.0;
  for (int i = 0; i < kPoints; ++i) {
    const Eigen::Vector3d p = Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized() * depth(rng);
    Eigen::Vector2d uv, uv_ref;
    Matrix23d Jp, Jp_ref;
    Matrix2Xd Jk, Jk_ref;
    const bool ok = cam_->project(p, uv, &Jp, &Jk) && cam_->inImage(uv);
    const bool ok_ref = ref_.project(p, uv_ref, &Jp_ref, &Jk_ref) && cam_->inImage(uv_ref);
    if (ok != ok_ref) {
      ++(ok ? only_ours : only_basalt);
      continue;
    }
    if (!ok) continue;
    ++compared;
    max_px = std::max(max_px, (uv - uv_ref).norm());
    max_J_point = std::max(max_J_point, (Jp - Jp_ref).norm() / Jp_ref.norm());
    max_J_params = std::max(max_J_params, (Jk - Jk_ref).norm() / Jk_ref.norm());
  }
  std::cout << "[ info ] compared " << compared << " in-image points; valid only in ours "
            << only_ours << ", only in Basalt " << only_basalt << "\n"
            << "[ info ] max pixel diff " << max_px << " px, J_point rel " << max_J_point
            << ", J_params rel " << max_J_params << "\n";
  EXPECT_GT(compared, 1000);
  EXPECT_LT(max_px, 1e-9);
  EXPECT_LT(max_J_point, 1e-9);
  EXPECT_LT(max_J_params, 1e-9);
}

TEST_P(BasaltCrossCheckTest, UnprojectionAgreesOnEveryPixel) {
  int compared = 0, only_ours = 0, only_basalt = 0;
  // Pixels only Basalt unprojects, where Basalt's own project then rejects the
  // bearing or misses the pixel by > 1e-6 px: not self-consistent in Basalt.
  int basalt_inconsistent = 0;
  double max_angle = 0.0;
  for (int v = 0; v < cam_->height(); ++v) {
    for (int u = 0; u < cam_->width(); ++u) {
      const Eigen::Vector2d uv(u, v);
      Eigen::Vector3d b, b_ref;
      const bool ok = cam_->unproject(uv, b);
      const bool ok_ref = ref_.unproject(uv, b_ref);
      if (ok != ok_ref) {
        ++(ok ? only_ours : only_basalt);
        Eigen::Vector2d uv_back;
        if (ok_ref && (!ref_.project(b_ref, uv_back, nullptr, nullptr) ||
                       (uv_back - uv).norm() > 1e-6)) {
          ++basalt_inconsistent;
        }
        continue;
      }
      if (!ok) continue;
      ++compared;
      max_angle = std::max(max_angle, std::atan2(b.cross(b_ref).norm(), b.dot(b_ref)));
    }
  }
  std::cout << "[ info ] compared " << compared << " pixels; valid only in ours " << only_ours
            << ", only in Basalt " << only_basalt << " (of which not self-consistent in Basalt "
            << basalt_inconsistent << "); max bearing diff " << max_angle << " rad\n";
  EXPECT_GT(compared, cam_->width() * cam_->height() / 2);
  EXPECT_LT(max_angle, 1e-9);
}

INSTANTIATE_TEST_SUITE_P(AllModels, BasaltCrossCheckTest, ::testing::ValuesIn(cameraCases()),
                         [](const auto& info) { return info.param.label; });

}  // namespace
}  // namespace msckf
