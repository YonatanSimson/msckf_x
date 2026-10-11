#include "msckf/calib/aprilgrid.h"

#include "msckf/vision/image.h"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
}

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msckf::calib {

Eigen::Vector3d AprilGrid::corner(int corner_id) const {
  const int tag = corner_id / 4, k = corner_id % 4;
  const int r = tag / cols, c = tag % cols;
  const double pitch = (1.0 + tag_spacing) * tag_size;
  const double x = c * pitch + ((k == 1 || k == 2) ? tag_size : 0.0);
  const double y = r * pitch + ((k == 2 || k == 3) ? tag_size : 0.0);
  return {x, y, 0.0};
}

AprilGrid AprilGrid::fromKalibrYaml(const std::string& path) {
  const YAML::Node n = YAML::LoadFile(path);
  if (n["target_type"].as<std::string>() != "aprilgrid") {
    throw std::runtime_error(path + ": target_type is not aprilgrid");
  }
  AprilGrid g;
  g.rows = n["tagRows"].as<int>();
  g.cols = n["tagCols"].as<int>();
  g.tag_size = n["tagSize"].as<double>();
  g.tag_spacing = n["tagSpacing"].as<double>();
  return g;
}

struct AprilGridDetector::Impl {
  apriltag_family_t* family = nullptr;
  apriltag_detector_t* detector = nullptr;
  ~Impl() {
    if (detector) apriltag_detector_destroy(detector);
    if (family) tag36h11_destroy(family);
  }
};

AprilGridDetector::AprilGridDetector(const AprilGrid& grid, const AprilGridDetectorOptions& options)
    : grid_(grid), options_(options), impl_(std::make_unique<Impl>()) {
  // tag36h11 is laid out as a 6x6 data block, a 1-bit black border and a
  // 1-bit white border (width_at_border 8, total_width 10). Kalibr prints a
  // 2-bit black border straight onto white paper, so the detected black edge
  // spans 10 bits and the data bits sit one bit further in.
  impl_->family = tag36h11_create();
  impl_->family->width_at_border = 10;
  impl_->family->total_width = 10;
  for (uint32_t i = 0; i < impl_->family->nbits; ++i) {
    impl_->family->bit_x[i] += 1;
    impl_->family->bit_y[i] += 1;
  }
  impl_->detector = apriltag_detector_create();
  apriltag_detector_add_family_bits(impl_->detector, impl_->family, 2);
  impl_->detector->quad_decimate = options_.quad_decimate;
  impl_->detector->nthreads = options_.threads;
  impl_->detector->refine_edges = true;
}

AprilGridDetector::~AprilGridDetector() = default;

GridObservation AprilGridDetector::detect(const cv::Mat& image_in) const {
  const cv::Mat image = vision::toGray8(image_in);
  // Each tag's outer black edge meets the grid's small corner squares at
  // X-junctions, so after thresholding the tag border and its four squares form
  // one black blob whose outline is not a quad, and apriltag (which fits quads
  // to blob outlines) misses the tag. Kalibr's gradient-based detector does not
  // have this problem. A 2x2 grayscale dilation shrinks black regions by about a
  // pixel, which opens the junctions while keeping the ~2.5 px data bits of a
  // 512 px fisheye view decodable. Corners are refined on the original image.
  cv::Mat opened;
  if (options_.open_junctions) {
    cv::dilate(image, opened, cv::Mat::ones(2, 2, CV_8U));
  } else {
    opened = image;
  }
  image_u8_t im{opened.cols, opened.rows, static_cast<int32_t>(opened.step), opened.data};
  zarray_t* detections = apriltag_detector_detect(impl_->detector, &im);

  struct Tag {
    int id;
    double p[4][2];
  };
  std::vector<Tag> tags;
  const double b = options_.min_border_distance;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t* det;
    zarray_get(detections, i, &det);
    if (det->id < 0 || det->id >= grid_.numTags()) continue;
    bool inside = true;
    for (const auto& p : det->p) {
      inside &= p[0] >= b && p[0] <= image.cols - b && p[1] >= b && p[1] <= image.rows - b;
    }
    if (!inside) continue;
    Tag t{det->id, {}};
    std::copy(&det->p[0][0], &det->p[0][0] + 8, &t.p[0][0]);
    tags.push_back(t);
  }
  apriltag_detections_destroy(detections);

  // A repeated id means a tag that is not on the board: drop every copy.
  std::sort(tags.begin(), tags.end(), [](const Tag& a, const Tag& c) { return a.id < c.id; });
  std::vector<Tag> unique;
  for (size_t i = 0; i < tags.size(); ++i) {
    const bool dup = (i > 0 && tags[i - 1].id == tags[i].id) ||
                     (i + 1 < tags.size() && tags[i + 1].id == tags[i].id);
    if (!dup) unique.push_back(tags[i]);
  }

  GridObservation obs;
  if (static_cast<int>(unique.size()) < options_.min_tags) return obs;

  // apriltag corners wrap counter-clockwise in the image starting at the tag's
  // bottom-left, which is Kalibr's corner order 0..3.
  std::vector<cv::Point2f> raw, refined;
  for (const Tag& t : unique) {
    for (const auto& p : t.p) raw.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]));
  }
  refined = raw;
  if (options_.subpix) {
    const int w = options_.subpix_half_window;
    cv::cornerSubPix(image, refined, cv::Size(w, w), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
  }
  obs.num_tags = static_cast<int>(unique.size());
  for (size_t t = 0; t < unique.size(); ++t) {
    for (int k = 0; k < 4; ++k) {
      const size_t i = 4 * t + k;
      const cv::Point2f d = refined[i] - raw[i];
      if (d.dot(d) > options_.max_subpix_displacement2) continue;
      obs.ids.push_back(4 * unique[t].id + k);
      obs.pixels.emplace_back(refined[i].x, refined[i].y);
    }
  }
  return obs;
}

}  // namespace msckf::calib
