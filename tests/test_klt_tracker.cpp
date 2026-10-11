// Minimal KLT tracker: bearing RANSAC precision/recall on wide-angle bearings
// (including rays past 90 degrees and pure rotation), and sub-pixel tracking
// of a texture under a known image motion.

#include <gtest/gtest.h>

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/pinhole_radtan.h"
#include "msckf/vision/bearing_ransac.h"
#include "msckf/vision/klt_tracker.h"
#include "synthetic_texture.h"

#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

#include <random>

namespace msckf::vision {
namespace {

struct BearingCase {
  std::vector<Eigen::Vector3d> a, b;
  std::vector<bool> outlier;
};

// Points all around camera a (rays past 90 degrees included), seen again
// from a camera moved by T_ab; a fraction replaced by random bearings.
BearingCase makeBearings(const Sophus::SE3d& T_ab, int n, double outlier_frac, double noise_rad,
                         uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  BearingCase c;
  const Sophus::SE3d T_ba = T_ab.inverse();
  auto jitter = [&](const Eigen::Vector3d& f) {
    return (f + noise_rad * Eigen::Vector3d(g(rng), g(rng), g(rng))).normalized();
  };
  while (static_cast<int>(c.a.size()) < n) {
    Eigen::Vector3d dir(g(rng), g(rng), g(rng));
    dir.normalize();
    if (dir.z() < -0.3) continue;  // up to ~107 degrees off axis
    const Eigen::Vector3d p_a = dir * (2.0 + 8.0 * u(rng));
    const Eigen::Vector3d p_b = T_ba * p_a;
    const bool out = u(rng) < outlier_frac;
    c.a.push_back(jitter(dir));
    c.b.push_back(out ? Eigen::Vector3d(g(rng), g(rng), g(rng)).normalized() : jitter(p_b.normalized()));
    c.outlier.push_back(out);
  }
  return c;
}

struct Rates {
  double precision, recall;
};

Rates rates(const BearingCase& c, const BearingRansacResult& r) {
  int tp = 0, fp = 0, fn = 0;
  for (size_t i = 0; i < c.a.size(); ++i) {
    const bool flagged = !r.inlier[i];
    tp += flagged && c.outlier[i];
    fp += flagged && !c.outlier[i];
    fn += !flagged && c.outlier[i];
  }
  return {tp + fp > 0 ? static_cast<double>(tp) / (tp + fp) : 1.0,
          tp + fn > 0 ? static_cast<double>(tp) / (tp + fn) : 1.0};
}

void expectPrecisionRecall(const BearingCase& c, const BearingRansacResult& r, double min_rate) {
  int tp = 0, fp = 0, fn = 0;  // positives = flagged as outlier
  for (size_t i = 0; i < c.a.size(); ++i) {
    const bool flagged = !r.inlier[i];
    tp += flagged && c.outlier[i];
    fp += flagged && !c.outlier[i];
    fn += !flagged && c.outlier[i];
  }
  const double precision = tp + fp > 0 ? static_cast<double>(tp) / (tp + fp) : 1.0;
  const double recall = tp + fn > 0 ? static_cast<double>(tp) / (tp + fn) : 1.0;
  std::cout << "[ info ] outlier precision " << precision << ", recall " << recall << " ("
            << r.iterations << " iterations)\n";
  EXPECT_GT(precision, min_rate);
  EXPECT_GT(recall, min_rate);
}

TEST(BearingRansac, RejectsOutliersWithWideFieldOfView) {
  const Sophus::SE3d T_ab(Sophus::SO3d::exp(Eigen::Vector3d(0.05, -0.1, 0.08)),
                          Eigen::Vector3d(0.3, 0.05, 0.1));
  const BearingCase c = makeBearings(T_ab, 300, 0.3, 2e-4, 5);
  BearingRansacConfig cfg;
  cfg.threshold_rad = 1e-3;
  const BearingRansacResult r = bearingRansac(c.a, c.b, cfg);
  ASSERT_TRUE(r.ok);
  expectPrecisionRecall(c, r, 0.95);
}

// Harder case: half outliers and noise close to the threshold. Averaged over
// seeds, LO-MSAC must classify better than plain RANSAC (and well overall).
TEST(BearingRansac, LoMsacBeatsPlainRansac) {
  const Sophus::SE3d T_ab(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.05, 0.04)),
                          Eigen::Vector3d(0.1, 0.02, 0.05));
  double f1[2] = {0.0, 0.0};
  constexpr int kSeeds = 20;
  for (int seed = 0; seed < kSeeds; ++seed) {
    const BearingCase c = makeBearings(T_ab, 200, 0.5, 4e-4, 100 + seed);
    for (int m = 0; m < 2; ++m) {
      BearingRansacConfig cfg;
      cfg.threshold_rad = 1.5e-3;
      cfg.method = m == 0 ? RobustMethod::kRansac : RobustMethod::kLoMsac;
      cfg.seed = seed + 1;
      const Rates r = rates(c, bearingRansac(c.a, c.b, cfg));
      f1[m] += 2.0 * r.precision * r.recall / (r.precision + r.recall) / kSeeds;
    }
  }
  std::cout << "[ info ] 50 % outliers, noise near threshold: mean F1 RANSAC " << f1[0]
            << ", LO-MSAC " << f1[1] << "\n";
  EXPECT_GT(f1[1], f1[0]);
  EXPECT_GT(f1[1], 0.95);
}

TEST(BearingRansac, KeepsInliersUnderPureRotation) {
  const Sophus::SE3d T_ab(Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.05, -0.2)), Eigen::Vector3d::Zero());
  const BearingCase c = makeBearings(T_ab, 200, 0.0, 2e-4, 6);
  const BearingRansacResult r = bearingRansac(c.a, c.b);
  ASSERT_TRUE(r.ok);
  EXPECT_GT(r.num_inliers, 0.95 * c.a.size());
}

using test::texture;

TEST(KltTracker, TracksKnownTranslationSubPixel) {
  constexpr int W = 640, H = 480;
  Eigen::VectorXd k(9);
  k << 400, 400, 320, 240, 0, 0, 0, 0, 0;
  auto camera = std::make_shared<PinholeRadtan>(W, H, k);
  const cv::Mat tex = texture(W + 200, H + 200, 3);
  KltTracker tracker(camera);

  // Frame n shows the texture shifted by n * (1.37, -0.61) px (sub-pixel,
  // bilinear), so a feature at uv in frame 0 sits at uv + n * shift.
  const Eigen::Vector2d shift(1.37, -0.61);
  std::map<int64_t, Eigen::Vector2d> origin;  // feature id -> uv minus its accumulated shift
  double sq = 0.0;
  int count = 0;
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
      // With pure translation nothing should be rejected by RANSAC.
      EXPECT_EQ(tracker.stats().rejected_ransac, 0) << n;
      EXPECT_GE(tracker.stats().tracked, 150) << n;
    }
  }
  const double rmse = std::sqrt(sq / count);
  std::cout << "[ info ] " << count << " tracked observations, RMSE " << rmse << " px\n";
  EXPECT_GT(count, 3000);
  EXPECT_LT(rmse, 0.05);
}

TEST(KltTracker, DetectsOnlyInsideTheValidRegion) {
  // Double Sphere with a circular image: corners of the sensor are invalid.
  Eigen::VectorXd k(6);
  k << 158.3, 158.3, 255.0, 256.9, -0.17, 0.59;
  auto camera = std::make_shared<DoubleSphere>(512, 512, k);
  KltTracker tracker(camera);
  const cv::Mat tex = texture(512, 512, 9);
  cv::Mat frame;
  tex.convertTo(frame, CV_8U);
  const auto& feats = tracker.process(frame);
  EXPECT_GT(feats.size(), 100u);
  for (const auto& f : feats) {
    EXPECT_TRUE(camera->validPixel(f.uv));
    EXPECT_NEAR(f.bearing.norm(), 1.0, 1e-12);
  }
}

}  // namespace
}  // namespace msckf::vision
