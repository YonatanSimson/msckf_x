#pragma once

#include "msckf/camera/camera_model.h"

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace msckf::vision {

struct TrackedFeature {
  int64_t id = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  Eigen::Vector3d bearing = Eigen::Vector3d::UnitZ();
  int age = 0;  // frames tracked since detection (0 in the detection frame)
  Eigen::Vector2d flow = Eigen::Vector2d::Zero();  // displacement over the last frame
};

// Why a feature was dropped in a frame.
enum class DropReason {
  kKltFailed,        // the alignment did not converge
  kLeftValidRegion,  // tracked to outside the image / camera model's valid region
  kForwardBackward,  // forward-backward round trip too far
  kRansac,           // epipolar outlier
};

struct DroppedFeature {
  TrackedFeature last;  // its state in the previous frame
  DropReason reason;
  // kForwardBackward: round-trip distance in px (NaN if the backward alignment
  // itself failed); kKltFailed: tracker-specific alignment error.
  double detail = 0.0;
};

struct TrackerFrameStats {
  int tracked = 0;           // features carried over from the previous frame
  int lost_klt = 0;          // alignment did not converge or left the valid region
  int rejected_fb = 0;       // forward-backward check
  int rejected_ransac = 0;   // epipolar outliers
  int detected = 0;          // new features
  double median_flow_px = 0.0;  // of the features carried over
  bool ransac_run = false;
  double time_ms = 0.0;
};

// Monocular sparse feature tracker on the raw (distorted) image. Owns the
// camera model's valid-region mask (eroded by a border) and the pixel-to-angle
// scale at the image centre, shared by every implementation.
class FeatureTracker {
 public:
  FeatureTracker(CameraModelPtr camera, int border_px);
  virtual ~FeatureTracker() = default;

  // Processes the next image (8 or 16 bit, grey or colour) and returns the
  // features alive in it.
  virtual const std::vector<TrackedFeature>& process(const cv::Mat& image) = 0;

  const std::vector<TrackedFeature>& features() const { return features_; }
  const TrackerFrameStats& stats() const { return stats_; }
  // Features dropped while processing the last image, with the reason.
  const std::vector<DroppedFeature>& dropped() const { return dropped_; }
  const cv::Mat& validMask() const { return valid_mask_; }
  double radiansPerPixel() const { return rad_per_px_; }

 protected:
  bool insideValid(const Eigen::Vector2d& uv) const;
  // Epipolar outlier rejection (LO-MSAC on bearings) of the features in
  // `kept` (whose previous state is features_[kept_src[i]]); drops the
  // outliers, recording them. Skipped below min_flow_px median flow.
  void rejectEpipolarOutliers(std::vector<TrackedFeature>& kept, const std::vector<size_t>& kept_src,
                              double threshold_px, double min_flow_px);
  // Fills stats_.median_flow_px and median_flow_ from the kept features.
  void updateMedianFlow(const std::vector<TrackedFeature>& kept);

  CameraModelPtr camera_;
  cv::Mat valid_mask_;  // 255 where the camera model unprojects, eroded by the border
  double rad_per_px_ = 0.0;
  std::vector<TrackedFeature> features_;
  TrackerFrameStats stats_;
  std::vector<DroppedFeature> dropped_;
  Eigen::Vector2d median_flow_ = Eigen::Vector2d::Zero();
  int64_t next_id_ = 0;
};

using FeatureTrackerPtr = std::unique_ptr<FeatureTracker>;

}  // namespace msckf::vision
