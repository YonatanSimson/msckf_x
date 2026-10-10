#pragma once

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace msckf::calib {

// Kalibr AprilGrid: rows x cols tag36h11 tags with a 2-bit black border, ids
// row-major from the bottom-left. Board frame (Kalibr's): origin at the
// bottom-left corner of tag 0, x to the right and y up as seen when facing the
// print, z = x cross y out of the printed face toward the viewer.
//
// Corner k of tag t has id 4 t + k, with k = 0..3 the tag's bottom-left,
// bottom-right, top-right, top-left corner (Kalibr's order).
struct AprilGrid {
  int rows = 6;
  int cols = 6;
  double tag_size = 0.088;  // black edge to black edge, m
  double tag_spacing = 0.3; // gap between tags as a fraction of tag_size

  int numTags() const { return rows * cols; }
  int numCorners() const { return 4 * numTags(); }
  Eigen::Vector3d corner(int corner_id) const;

  static AprilGrid fromKalibrYaml(const std::string& path);
};

// Corners of one image. ids index AprilGrid::corner.
struct GridObservation {
  std::vector<int> ids;
  std::vector<Eigen::Vector2d> pixels;
  int num_tags = 0;
  size_t size() const { return ids.size(); }
};

struct AprilGridDetectorOptions {
  // Kalibr defaults: tags with a corner closer than this to the image border
  // are dropped; corners moving more than sqrt(max_subpix_displacement2) px in
  // cv::cornerSubPix are dropped; images with fewer tags are rejected.
  double min_border_distance = 4.0;
  double max_subpix_displacement2 = 1.5;
  int min_tags = 4;
  bool subpix = true;
  bool open_junctions = true;  // see AprilGridDetector::detect
  int subpix_half_window = 2;
  float quad_decimate = 1.0f;
  int threads = 1;
};

// AprilTag (AprilRobotics, BSD-2) quad detection and decoding with a
// tag36h11 family re-described for a 2-bit black border, followed by Kalibr's
// cornerSubPix refinement on the tag corners, which on an AprilGrid are
// checkerboard X-junctions with the small corner squares.
class AprilGridDetector {
 public:
  explicit AprilGridDetector(const AprilGrid& grid, const AprilGridDetectorOptions& options = {});
  ~AprilGridDetector();
  AprilGridDetector(const AprilGridDetector&) = delete;
  AprilGridDetector& operator=(const AprilGridDetector&) = delete;

  // image: 8-bit single channel. Returns an empty observation when fewer than
  // min_tags tags pass the filters.
  GridObservation detect(const cv::Mat& image) const;

  const AprilGrid& grid() const { return grid_; }

 private:
  struct Impl;
  AprilGrid grid_;
  AprilGridDetectorOptions options_;
  std::unique_ptr<Impl> impl_;
};

// Converts 16-bit images to 8 bits (high byte); 8-bit images pass through.
cv::Mat toGray8(const cv::Mat& image);

}  // namespace msckf::calib
