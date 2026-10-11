// Vendored Basalt optical flow (BasaltFlowTracker): sub-pixel tracking of a
// texture under a known image motion, for both flow types, and detection
// restricted to the camera model's valid region when asked. Agreement with
// the unmodified Basalt on real sequences is checked by
// python/basalt_flow_check.py.

#include <gtest/gtest.h>

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/pinhole_radtan.h"
#include "msckf/vision/basalt_flow.h"
#include "synthetic_texture.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace msckf::vision {
namespace {

using test::texture;

class BasaltFlowShift : public ::testing::TestWithParam<BasaltFlowConfig::Type> {};

TEST_P(BasaltFlowShift, TracksKnownTranslationSubPixel) {
  constexpr int W = 640, H = 480;
  Eigen::VectorXd k(9);
  k << 400, 400, 320, 240, 0, 0, 0, 0, 0;
  auto camera = std::make_shared<PinholeRadtan>(W, H, k);
  const cv::Mat tex = texture(W + 200, H + 200, 3);
  BasaltFlowConfig cfg;
  cfg.type = GetParam();
  BasaltFlowTracker tracker(camera, cfg);

  // Frame n shows the texture shifted by n * (1.37, -0.61) px (sub-pixel,
  // bilinear), so a feature at uv in frame 0 sits at uv + n * shift.
  const Eigen::Vector2d shift(1.37, -0.61);
  std::map<int64_t, Eigen::Vector2d> origin;  // feature id -> uv minus its accumulated shift
  double sq = 0.0;
  int count = 0, interior_drops = 0;
  for (int n = 0; n < 30; ++n) {
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, n * shift.x() - 100, 0, 1, n * shift.y() - 100);
    cv::Mat frame;
    cv::warpAffine(tex, frame, M, cv::Size(W, H), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    frame.convertTo(frame, CV_8U);
    const auto& feats = tracker.process(frame);
    for (const auto& f : feats) {
      if (f.age == 0) {
        origin[f.id] = f.uv - n * shift;
        continue;
      }
      const Eigen::Vector2d e = f.uv - (origin.at(f.id) + n * shift);
      sq += e.squaredNorm();
      ++count;
    }
    if (n > 0) {
      // One feature per 50 px cell: 12 x 9 cells, minus Basalt's 19 px edge margin.
      EXPECT_GE(tracker.stats().tracked, 70) << n;
      // Points leave through the image edges as the texture shifts; away
      // from them, only weak corners should fail (forward-backward).
      for (const auto& d : tracker.dropped()) {
        const Eigen::Vector2d& p = d.last.uv;
        interior_drops += std::min({p.x(), p.y(), W - p.x(), H - p.y()}) > 25.0;
      }
    }
  }
  const double rmse = std::sqrt(sq / count);
  std::cout << "[ info ] " << (GetParam() == BasaltFlowConfig::Type::kPatch ? "patch" : "frame_to_frame") << ": "
            << count << " tracked observations, RMSE " << rmse << " px, " << interior_drops
            << " dropped away from the edges\n";
  EXPECT_GT(count, 2000);
  EXPECT_LT(interior_drops, 0.02 * count);
  EXPECT_LT(rmse, 0.05);
}

INSTANTIATE_TEST_SUITE_P(Types, BasaltFlowShift,
                         ::testing::Values(BasaltFlowConfig::Type::kFrameToFrame, BasaltFlowConfig::Type::kPatch),
                         [](const auto& info) {
                           return info.param == BasaltFlowConfig::Type::kPatch ? "Patch" : "FrameToFrame";
                         });

TEST(BasaltFlow, MasksTheValidRegionWhenAsked) {
  // Double Sphere with a short focal length: the sensor corners lie outside
  // the model's domain (r^2 > 1 / (2 alpha - 1) for alpha > 0.5).
  Eigen::VectorXd k(6);
  k << 120.0, 120.0, 255.0, 256.9, -0.17, 0.59;
  auto camera = std::make_shared<DoubleSphere>(512, 512, k);
  const cv::Mat tex = texture(512, 512, 9);
  cv::Mat frame;
  tex.convertTo(frame, CV_8U);

  BasaltFlowConfig cfg;
  cfg.grid_size = 30;
  cfg.mask_valid_region = true;
  BasaltFlowTracker masked(camera, cfg);
  const auto& feats = masked.process(frame);
  EXPECT_GT(feats.size(), 100u);
  for (const auto& f : feats) {
    EXPECT_TRUE(camera->validPixel(f.uv));
    EXPECT_NEAR(f.bearing.norm(), 1.0, 1e-12);
  }

  // Basalt itself detects everywhere the texture has corners, here also in
  // the invalid sensor corners, whose bearings are then NaN.
  cfg.mask_valid_region = false;
  BasaltFlowTracker unmasked(camera, cfg);
  int outside = 0;
  for (const auto& f : unmasked.process(frame)) {
    if (!camera->validPixel(f.uv)) {
      ++outside;
      EXPECT_TRUE(f.bearing.array().isNaN().all());
    }
  }
  ASSERT_FALSE(camera->validPixel(Eigen::Vector2d(5, 5)));
  EXPECT_GT(outside, 0);
}

}  // namespace
}  // namespace msckf::vision
