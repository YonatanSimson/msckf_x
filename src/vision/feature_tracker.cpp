#include "msckf/vision/feature_tracker.h"

#include "msckf/vision/bearing_ransac.h"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msckf::vision {

FeatureTracker::FeatureTracker(CameraModelPtr camera, int border_px) : camera_(std::move(camera)) {
  const int w = camera_->width(), h = camera_->height();
  cv::Mat valid(h, w, CV_8U, cv::Scalar(0));
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      if (camera_->validPixel(Eigen::Vector2d(u, v))) valid.at<uint8_t>(v, u) = 255;
    }
  }
  const int b = border_px;
  cv::rectangle(valid, cv::Rect(0, 0, w, h), cv::Scalar(0), 2 * b);  // image border
  cv::erode(valid, valid_mask_, cv::Mat(), cv::Point(-1, -1), b);   // valid-region border
  // Angle subtended by one pixel at the image center, to express pixel
  // thresholds as angles for any camera model.
  Eigen::Vector3d b0, b1;
  const Eigen::Vector2d c(0.5 * w, 0.5 * h);
  if (!camera_->unproject(c, b0) || !camera_->unproject(c + Eigen::Vector2d(1.0, 0.0), b1)) {
    throw std::invalid_argument("FeatureTracker: image center does not unproject");
  }
  rad_per_px_ = std::atan2(b0.cross(b1).norm(), b0.dot(b1));
}

bool FeatureTracker::insideValid(const Eigen::Vector2d& uv) const {
  const int u = static_cast<int>(std::lround(uv.x())), v = static_cast<int>(std::lround(uv.y()));
  return u >= 0 && v >= 0 && u < valid_mask_.cols && v < valid_mask_.rows &&
         valid_mask_.at<uint8_t>(v, u) != 0;
}

void FeatureTracker::updateMedianFlow(const std::vector<TrackedFeature>& kept) {
  if (kept.empty()) return;
  std::vector<double> norm, fx, fy;
  for (const auto& f : kept) {
    norm.push_back(f.flow.norm());
    fx.push_back(f.flow.x());
    fy.push_back(f.flow.y());
  }
  const size_t m = kept.size() / 2;
  std::nth_element(norm.begin(), norm.begin() + m, norm.end());
  std::nth_element(fx.begin(), fx.begin() + m, fx.end());
  std::nth_element(fy.begin(), fy.begin() + m, fy.end());
  stats_.median_flow_px = norm[m];
  // Median displacement vector, the starting guess for new features.
  median_flow_ = Eigen::Vector2d(fx[m], fy[m]);
}

void FeatureTracker::rejectEpipolarOutliers(std::vector<TrackedFeature>& kept,
                                            const std::vector<size_t>& kept_src, double threshold_px,
                                            double min_flow_px) {
  if (kept.size() < 8 || stats_.median_flow_px < min_flow_px) return;
  std::vector<Eigen::Vector3d> f_prev, f_next;
  for (size_t i = 0; i < kept.size(); ++i) {
    f_prev.push_back(features_[kept_src[i]].bearing);
    f_next.push_back(kept[i].bearing);
  }
  BearingRansacConfig rc;
  rc.threshold_rad = threshold_px * rad_per_px_;
  const BearingRansacResult r = bearingRansac(f_prev, f_next, rc);
  stats_.ransac_run = r.ok;
  if (!r.ok) return;
  std::vector<TrackedFeature> inliers;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (r.inlier[i]) {
      inliers.push_back(kept[i]);
    } else {
      dropped_.push_back({features_[kept_src[i]], DropReason::kRansac});
    }
  }
  stats_.rejected_ransac = static_cast<int>(kept.size() - inliers.size());
  kept.swap(inliers);
}

}  // namespace msckf::vision
