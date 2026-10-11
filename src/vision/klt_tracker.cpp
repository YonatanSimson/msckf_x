#include "msckf/vision/klt_tracker.h"

#include "msckf/vision/image.h"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace msckf::vision {

KltTracker::KltTracker(CameraModelPtr camera, const KltTrackerConfig& config)
    : FeatureTracker(std::move(camera), config.border_px), config_(config) {}

const std::vector<TrackedFeature>& KltTracker::process(const cv::Mat& image_in) {
  const auto t_start = std::chrono::steady_clock::now();
  const cv::Mat image = toGray8(image_in);
  if (image.cols != camera_->width() || image.rows != camera_->height()) {
    throw std::invalid_argument("KltTracker: image size does not match the camera model");
  }
  stats_ = TrackerFrameStats{};
  dropped_.clear();
  std::vector<cv::Mat> pyramid;
  const cv::Size win(config_.window_px, config_.window_px);
  cv::buildOpticalFlowPyramid(image, pyramid, win, config_.pyramid_levels);

  if (!features_.empty() && !prev_pyramid_.empty()) {
    std::vector<cv::Point2f> prev, next, back;
    for (const auto& f : features_) {
      prev.emplace_back(f.uv.x(), f.uv.y());
      const Eigen::Vector2d guess = config_.constant_velocity ? Eigen::Vector2d(f.uv + f.flow) : f.uv;
      next.emplace_back(guess.x(), guess.y());
    }
    std::vector<uint8_t> st_fwd, st_bwd;
    std::vector<float> err, err_fwd;
    const cv::TermCriteria term(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_pyramid_, pyramid, prev, next, st_fwd, err_fwd, win,
                             config_.pyramid_levels, term,
                             config_.constant_velocity ? cv::OPTFLOW_USE_INITIAL_FLOW : 0);
    back = prev;
    cv::calcOpticalFlowPyrLK(pyramid, prev_pyramid_, next, back, st_bwd, err, win,
                             config_.pyramid_levels, term, cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<TrackedFeature> kept;
    std::vector<size_t> kept_src;  // index into features_ (state in the previous frame)
    for (size_t i = 0; i < features_.size(); ++i) {
      const cv::Point2f& p = next[i];
      const bool inside = insideValid(Eigen::Vector2d(p.x, p.y));
      if (!st_fwd[i] || !inside) {
        ++stats_.lost_klt;
        dropped_.push_back({features_[i], st_fwd[i] ? DropReason::kLeftValidRegion : DropReason::kKltFailed,
                            st_fwd[i] ? 0.0 : static_cast<double>(err_fwd[i])});
        continue;
      }
      const cv::Point2f d = back[i] - prev[i];
      if (!st_bwd[i] || d.dot(d) > config_.fb_threshold_px * config_.fb_threshold_px) {
        ++stats_.rejected_fb;
        dropped_.push_back({features_[i], DropReason::kForwardBackward,
                            st_bwd[i] ? std::sqrt(static_cast<double>(d.dot(d)))
                                      : std::numeric_limits<double>::quiet_NaN()});
        continue;
      }
      TrackedFeature f = features_[i];
      f.uv = Eigen::Vector2d(p.x, p.y);
      f.flow = f.uv - features_[i].uv;
      if (!camera_->unproject(f.uv, f.bearing)) {
        ++stats_.lost_klt;
        dropped_.push_back({features_[i], DropReason::kLeftValidRegion});
        continue;
      }
      ++f.age;
      kept.push_back(f);
      kept_src.push_back(i);
    }

    updateMedianFlow(kept);
    if (config_.ransac) {
      rejectEpipolarOutliers(kept, kept_src, config_.ransac_threshold_px, config_.ransac_min_flow_px);
    }
    features_.swap(kept);
    stats_.tracked = static_cast<int>(features_.size());
  }

  detect(image);
  prev_pyramid_ = std::move(pyramid);
  stats_.time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
  return features_;
}

void KltTracker::detect(const cv::Mat& image) {
  const int need = config_.max_features - static_cast<int>(features_.size());
  if (need <= 0) return;

  // Mask: valid region minus a disc around every live feature.
  cv::Mat mask = valid_mask_.clone();
  const int r = static_cast<int>(std::lround(config_.min_distance_px));
  std::vector<int> cell_count(config_.grid_cols * config_.grid_rows, 0);
  const double cw = static_cast<double>(image.cols) / config_.grid_cols;
  const double ch = static_cast<double>(image.rows) / config_.grid_rows;
  auto cellOf = [&](double u, double v) {
    const int cx = std::clamp(static_cast<int>(u / cw), 0, config_.grid_cols - 1);
    const int cy = std::clamp(static_cast<int>(v / ch), 0, config_.grid_rows - 1);
    return cy * config_.grid_cols + cx;
  };
  for (const auto& f : features_) {
    cv::circle(mask, cv::Point(static_cast<int>(f.uv.x()), static_cast<int>(f.uv.y())), r,
               cv::Scalar(0), cv::FILLED);
    ++cell_count[cellOf(f.uv.x(), f.uv.y())];
  }

  // Candidates sorted by corner strength, accepted while their grid cell is
  // below its share, so detections spread over the image.
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(image, corners, 4 * need, config_.quality_level, config_.min_distance_px,
                          mask, config_.block_size);
  const int cells = config_.grid_cols * config_.grid_rows;
  const int cap = (config_.max_features + cells - 1) / cells;
  for (int pass = 0; pass < 2 && stats_.detected < need; ++pass) {
    for (auto& c : corners) {
      if (stats_.detected >= need) break;
      if (c.x < 0) continue;  // taken in the first pass
      const int cell = cellOf(c.x, c.y);
      // First pass respects the per-cell cap; the second fills the remainder.
      if (pass == 0 && cell_count[cell] >= cap) continue;
      TrackedFeature f;
      f.uv = Eigen::Vector2d(c.x, c.y);
      f.flow = median_flow_;
      if (!camera_->unproject(f.uv, f.bearing)) continue;
      f.id = next_id_++;
      features_.push_back(f);
      ++cell_count[cell];
      ++stats_.detected;
      c.x = -1.0f;
    }
  }
}

}  // namespace msckf::vision
