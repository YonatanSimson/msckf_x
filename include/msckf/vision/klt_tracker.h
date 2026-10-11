#pragma once

#include "msckf/vision/feature_tracker.h"

namespace msckf::vision {

// Minimal KLT tracker for Stage 1 (plan, "Minimal KLT tracker"): OpenCV
// pyramidal Lucas-Kanade on the raw (distorted) image, a forward-backward
// check, RANSAC on unit bearings between consecutive frames, and Shi-Tomasi
// detection balanced over a grid and masked by the camera model's valid
// region. No gyro prediction; the Stage 2 front end replaces this.
struct KltTrackerConfig {
  int max_features = 200;
  int grid_cols = 8;
  int grid_rows = 8;
  double min_distance_px = 15.0;  // between features
  double quality_level = 0.01;    // Shi-Tomasi, relative to the best corner
  int block_size = 3;             // Shi-Tomasi structure-tensor window (px)
  int border_px = 8;              // margin to the image border and valid region
  int window_px = 21;             // LK window
  int pyramid_levels = 3;         // levels above the full-resolution image
  double fb_threshold_px = 0.5;   // forward-backward round trip
  // Start each feature's LK search at its previous position plus its last
  // displacement (new features: the median displacement of the tracked ones).
  // Without an IMU this is the cheap way to extend the capture range; the
  // Stage 2 front end adds gyro-aided prediction.
  bool constant_velocity = false;
  bool ransac = true;
  double ransac_threshold_px = 1.0;  // converted to an angle at the image center
  // RANSAC is skipped when the median flow is below this (no geometry to fit).
  double ransac_min_flow_px = 0.5;
};

class KltTracker : public FeatureTracker {
 public:
  explicit KltTracker(CameraModelPtr camera, const KltTrackerConfig& config = {});

  const std::vector<TrackedFeature>& process(const cv::Mat& image) override;

 private:
  void detect(const cv::Mat& image);

  KltTrackerConfig config_;
  std::vector<cv::Mat> prev_pyramid_;
};

}  // namespace msckf::vision
