#pragma once

#include "msckf/vision/feature_tracker.h"

#include <memory>

namespace msckf::vision {

// Basalt's optical flow front end (Usenko et al., BSD-3; vendored in
// src/vision/basalt_flow.cpp, see the license there), adopted as our tracker
// (plan, Stage 1, "Head-to-head against Basalt"). Per frame:
//
//  1. Track: every live point is aligned into the new image by
//     inverse-compositional Gauss-Newton on an SE2 warp of a sparse patch
//     (mean-normalized intensities), coarse to fine over the pyramid,
//     starting at its previous position. kFrameToFrame aligns the patch of
//     the previous frame; kPatch the patch of the detection frame.
//  2. Forward-backward: the point is aligned back; kept if the round trip
//     ends within sqrt(max_recovered_dist2) px of where it started.
//  3. Detect: one FAST corner in every grid cell without a live point.
//
// Defaults are Basalt's VioConfig defaults; Basalt's shipped EuRoC and TUM-VI
// configs use max_recovered_dist2 = 0.04 and grid 50 / 40 px. With the
// defaults (no valid-region mask) the output equals Basalt's own (gated by
// python/basalt_flow_check.py against the unmodified Basalt module).
struct BasaltFlowConfig {
  enum class Type { kFrameToFrame, kPatch };
  Type type = Type::kFrameToFrame;
  int grid_size = 50;                // detection cell, px
  float max_recovered_dist2 = 0.09f;  // forward-backward, px^2
  int pattern = 51;                  // 24, 50, 51 or 52 (Basalt's patterns)
  int max_iterations = 5;            // Gauss-Newton iterations per level
  int levels = 3;                    // pyramid levels above full resolution
  // Not in Basalt: drop points outside the camera model's valid region
  // (eroded by border_px) and do not detect there.
  bool mask_valid_region = false;
  int border_px = 8;
};

class BasaltFlowTracker : public FeatureTracker {
 public:
  explicit BasaltFlowTracker(CameraModelPtr camera, const BasaltFlowConfig& config = {});
  ~BasaltFlowTracker() override;

  // 8- or 16-bit, grey or colour. 8-bit values are shifted to the high byte,
  // as Basalt's dataset readers do. A feature's bearing is NaN when the camera
  // model cannot unproject its pixel (only without mask_valid_region).
  const std::vector<TrackedFeature>& process(const cv::Mat& image) override;

  struct Impl;

 private:
  BasaltFlowConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msckf::vision
