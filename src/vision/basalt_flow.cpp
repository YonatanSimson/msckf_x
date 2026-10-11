// Basalt's optical flow front end, vendored from Basalt
// (https://gitlab.com/VladyslavUsenko/basalt, local checkout ~/work/basalt at
// 914808d): include/basalt/optical_flow/{patterns.h, patch.h,
// frame_to_frame_optical_flow.h, patch_optical_flow.h} and detectKeypoints()
// from src/utils/keypoints.cpp. Changes: one camera only (no stereo matching or
// epipolar filter), no processing thread or queues, the TBB parallel loops run
// serially (each point is independent, so the result is unchanged), patches
// of dropped points are freed, and drop reasons and an optional valid-region
// mask are added for our FeatureTracker interface. The image and pyramid come
// from basalt-headers.
//
// Original license:
//
// BSD 3-Clause License
//
// This file is part of the Basalt project.
// https://gitlab.com/VladyslavUsenko/basalt.git
//
// Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "msckf/vision/basalt_flow.h"

#include <basalt/image/image_pyr.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se2.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace msckf::vision {

namespace {

// ------------------------------------------------------------ patterns.h ----

template <class Scalar>
struct Pattern24 {
  static constexpr Scalar pattern_raw[][2] = {
      {-1, 5},  {1, 5},                                                  //
      {-3, 3},  {-1, 3},  {1, 3},   {3, 3},                              //
      {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},  {5, 1},            //
      {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1}, {5, -1},           //
      {-3, -3}, {-1, -3}, {1, -3},  {3, -3},                             //
      {-1, -5}, {1, -5}};
  static constexpr int PATTERN_SIZE = sizeof(pattern_raw) / (2 * sizeof(Scalar));
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern24<Scalar>::Matrix2P Pattern24<Scalar>::pattern2 =
    Eigen::Map<const typename Pattern24<Scalar>::Matrix2P>(&Pattern24<Scalar>::pattern_raw[0][0]);

template <class Scalar>
struct Pattern52 {
  static constexpr Scalar pattern_raw[][2] = {
      {-3, 7},  {-1, 7},  {1, 7},   {3, 7},                                                  //
      {-5, 5},  {-3, 5},  {-1, 5},  {1, 5},   {3, 5},  {5, 5},                               //
      {-7, 3},  {-5, 3},  {-3, 3},  {-1, 3},  {1, 3},  {3, 3},  {5, 3},  {7, 3},             //
      {-7, 1},  {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},  {5, 1},  {7, 1},             //
      {-7, -1}, {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1}, {5, -1}, {7, -1},            //
      {-7, -3}, {-5, -3}, {-3, -3}, {-1, -3}, {1, -3}, {3, -3}, {5, -3}, {7, -3},            //
      {-5, -5}, {-3, -5}, {-1, -5}, {1, -5},  {3, -5}, {5, -5},                              //
      {-3, -7}, {-1, -7}, {1, -7},  {3, -7}};
  static constexpr int PATTERN_SIZE = sizeof(pattern_raw) / (2 * sizeof(Scalar));
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern52<Scalar>::Matrix2P Pattern52<Scalar>::pattern2 =
    Eigen::Map<const typename Pattern52<Scalar>::Matrix2P>(&Pattern52<Scalar>::pattern_raw[0][0]);

// Same as Pattern52 but twice smaller.
template <class Scalar>
struct Pattern51 {
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern51<Scalar>::Matrix2P Pattern51<Scalar>::pattern2 = 0.5 * Pattern52<Scalar>::pattern2;

// Same as Pattern52 but 0.75 smaller.
template <class Scalar>
struct Pattern50 {
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern50<Scalar>::Matrix2P Pattern50<Scalar>::pattern2 = 0.75 * Pattern52<Scalar>::pattern2;

// --------------------------------------------------------------- patch.h ----

using basalt::Image;

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() = default;

  OpticalFlowPatch(const Image<const uint16_t>& img, const Vector2& pos) { setFromImage(img, pos); }

  template <typename ImgT>
  static void setDataJacSe2(const ImgT& img, const Vector2& pos, Scalar& mean, VectorP& data, MatrixP3& J_se2) {
    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2;
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p = pos + pattern2.col(i);

      // Fill jacobians with respect to SE2 warp
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);

      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        sum += valGrad[0];
        J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        grad_sum_se2 += J_se2.row(i);
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;

    const Scalar mean_inv = num_valid_points / sum;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) {
        J_se2.row(i) -= grad_sum_se2.transpose() * data[i] / sum;
        data[i] *= mean_inv;
      } else {
        J_se2.row(i).setZero();
      }
    }
    J_se2 *= mean_inv;
  }

  void setFromImage(const Image<const uint16_t>& img, const Vector2& pos_in) {
    this->pos = pos_in;

    MatrixP3 J_se2;

    setDataJacSe2(img, pos, mean, data, J_se2);

    Matrix3 H_se2 = J_se2.transpose() * J_se2;
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv);

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose();

    // An all-black patch cannot be normalized: mean "zero", NaN/inf below.
    valid = mean > std::numeric_limits<Scalar>::epsilon() && H_se2_inv_J_se2_T.array().isFinite().all() &&
            data.array().isFinite().all();
  }

  inline bool residual(const Image<const uint16_t>& img, const Matrix2P& transformed_pattern,
                       VectorP& residual) const {
    Scalar sum = 0;
    int num_valid_points = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (img.InBounds(transformed_pattern.col(i), 2)) {
        residual[i] = img.interp<Scalar>(transformed_pattern.col(i));
        sum += residual[i];
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }

    // all-black patch cannot be normalized
    if (sum < std::numeric_limits<Scalar>::epsilon()) {
      residual.setZero();
      return false;
    }

    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i];
        num_residuals++;
      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;
  }

  Vector2 pos = Vector2::Zero();
  VectorP data = VectorP::Zero();  // negative if the point is not valid
  Matrix3P H_se2_inv_J_se2_T = Matrix3P::Zero();
  Scalar mean = 0;
  bool valid = false;
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P OpticalFlowPatch<Scalar, Pattern>::pattern2 =
    Pattern::pattern2;

// ------------------------------------------------ keypoints.cpp (detect) ----

constexpr int EDGE_THRESHOLD = 19;

// One FAST corner (threshold 40, halved down to 5) per grid cell holding no
// current point. accept() is ours: points it rejects are skipped (Basalt
// accepts everything).
template <typename Accept>
std::vector<Eigen::Vector2d> detectKeypoints(const Image<const uint16_t>& img_raw, int PATCH_SIZE,
                                             int num_points_cell,
                                             const std::vector<Eigen::Vector2d>& current_points, Accept accept) {
  std::vector<Eigen::Vector2d> corners;

  const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
  const size_t x_stop = x_start + PATCH_SIZE * (img_raw.w / PATCH_SIZE - 1);

  const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
  const size_t y_stop = y_start + PATCH_SIZE * (img_raw.h / PATCH_SIZE - 1);

  Eigen::MatrixXi cells;
  cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

  for (const Eigen::Vector2d& p : current_points) {
    if (p[0] >= x_start && p[1] >= y_start && p[0] < x_stop + PATCH_SIZE && p[1] < y_stop + PATCH_SIZE) {
      int x = (p[0] - x_start) / PATCH_SIZE;
      int y = (p[1] - y_start) / PATCH_SIZE;

      cells(y, x) += 1;
    }
  }

  for (size_t x = x_start; x <= x_stop; x += PATCH_SIZE) {
    for (size_t y = y_start; y <= y_stop; y += PATCH_SIZE) {
      if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0) continue;

      const Image<const uint16_t> sub_img_raw = img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

      cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

      for (int yy = 0; yy < PATCH_SIZE; yy++) {
        uchar* sub_ptr = subImg.ptr(yy);
        for (int xx = 0; xx < PATCH_SIZE; xx++) {
          sub_ptr[xx] = (sub_img_raw(xx, yy) >> 8);
        }
      }

      int points_added = 0;
      int threshold = 40;

      while (points_added < num_points_cell && threshold >= 5) {
        std::vector<cv::KeyPoint> points;
        cv::FAST(subImg, points, threshold);

        std::sort(points.begin(), points.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool { return a.response > b.response; });

        for (size_t i = 0; i < points.size() && points_added < num_points_cell; i++) {
          const Eigen::Vector2d c(x + points[i].pt.x, y + points[i].pt.y);
          if (img_raw.InBounds(c.x(), c.y(), EDGE_THRESHOLD) && accept(c)) {
            corners.push_back(c);
            points_added++;
          }
        }

        threshold /= 2;
      }
    }
  }
  return corners;
}

// -------------------------- frame_to_frame / patch optical flow (1 cam) ----

using Pyramid = basalt::ManagedImagePyr<uint16_t>;
using KeypointId = int64_t;

enum class TrackResult { kOk, kFailed, kForwardBackward };

struct FlowStep {
  std::map<KeypointId, Eigen::AffineCompact2f> tracked;
  std::map<KeypointId, std::pair<TrackResult, float>> failed;  // reason, round-trip px
};

class FlowImplBase {
 public:
  virtual ~FlowImplBase() = default;
  // Tracks `prev` (transforms in old_pyr) into pyr.
  virtual FlowStep track(const Pyramid& old_pyr, const Pyramid& pyr,
                         const std::map<KeypointId, Eigen::AffineCompact2f>& prev) = 0;
  // Called for each new point (kPatch stores its reference patches).
  virtual void addPoint(const Pyramid& pyr, KeypointId id, const Eigen::Vector2f& pos) = 0;
  virtual void erase(KeypointId id) = 0;
};

template <typename Scalar, template <typename> typename Pattern>
class FlowImpl : public FlowImplBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Sophus::SE2<Scalar> SE2;

  explicit FlowImpl(const BasaltFlowConfig& config) : config(config) {}

  FlowStep track(const Pyramid& pyr_1, const Pyramid& pyr_2,
                 const std::map<KeypointId, Eigen::AffineCompact2f>& transform_map_1) override {
    FlowStep out;
    for (const auto& [id, transform_1] : transform_map_1) {
      Eigen::AffineCompact2f transform_2 = transform_1;

      bool valid = patch_mode() ? trackPointPatch(pyr_2, patches.at(id), transform_2)
                                : trackPoint(pyr_1, pyr_2, transform_1, transform_2);
      if (!valid) {
        out.failed[id] = {TrackResult::kFailed, std::numeric_limits<float>::quiet_NaN()};
        continue;
      }
      Eigen::AffineCompact2f transform_1_recovered = transform_2;

      valid = patch_mode() ? trackPointPatch(pyr_1, patches.at(id), transform_1_recovered)
                           : trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);
      if (!valid) {
        out.failed[id] = {TrackResult::kForwardBackward, std::numeric_limits<float>::quiet_NaN()};
        continue;
      }
      const Scalar dist2 = (transform_1.translation() - transform_1_recovered.translation()).squaredNorm();

      if (dist2 < config.max_recovered_dist2) {
        out.tracked[id] = transform_2;
      } else {
        out.failed[id] = {TrackResult::kForwardBackward, std::sqrt(dist2)};
      }
    }
    return out;
  }

  void addPoint(const Pyramid& pyr, KeypointId id, const Eigen::Vector2f& pos) override {
    if (!patch_mode()) return;
    std::vector<PatchT, Eigen::aligned_allocator<PatchT>>& p = patches[id];
    for (int l = 0; l <= config.levels; l++) {
      Scalar scale = 1 << l;
      Vector2 pos_scaled = pos.cast<Scalar>() / scale;
      p.emplace_back(pyr.lvl(l), pos_scaled);
    }
  }

  void erase(KeypointId id) override { patches.erase(id); }

 private:
  bool patch_mode() const { return config.type == BasaltFlowConfig::Type::kPatch; }

  // FrameToFrameOpticalFlow::trackPoint: the patch is taken from the old image.
  inline bool trackPoint(const Pyramid& old_pyr, const Pyramid& pyr, const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  // PatchOpticalFlow::trackPoint: the patches are those of the detection frame.
  inline bool trackPointPatch(const Pyramid& pyr, const std::vector<PatchT, Eigen::aligned_allocator<PatchT>>& patch_vec,
                              Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int level = config.levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      const auto& p = patch_vec[level];
      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2, const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0; patch_valid && iteration < config.max_iterations; iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat = transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  BasaltFlowConfig config;
  std::map<KeypointId, std::vector<PatchT, Eigen::aligned_allocator<PatchT>>> patches;
};

std::unique_ptr<FlowImplBase> makeFlow(const BasaltFlowConfig& c) {
  switch (c.pattern) {
    case 24: return std::make_unique<FlowImpl<float, Pattern24>>(c);
    case 50: return std::make_unique<FlowImpl<float, Pattern50>>(c);
    case 51: return std::make_unique<FlowImpl<float, Pattern51>>(c);
    case 52: return std::make_unique<FlowImpl<float, Pattern52>>(c);
  }
  throw std::invalid_argument("BasaltFlowTracker: unsupported pattern " + std::to_string(c.pattern));
}

}  // namespace

struct BasaltFlowTracker::Impl {
  std::unique_ptr<FlowImplBase> flow;
  std::unique_ptr<Pyramid> pyramid, old_pyramid;
  std::map<KeypointId, Eigen::AffineCompact2f> transforms;
};

BasaltFlowTracker::BasaltFlowTracker(CameraModelPtr camera, const BasaltFlowConfig& config)
    : FeatureTracker(std::move(camera), config.border_px), config_(config), impl_(std::make_unique<Impl>()) {
  impl_->flow = makeFlow(config_);
}

BasaltFlowTracker::~BasaltFlowTracker() = default;

const std::vector<TrackedFeature>& BasaltFlowTracker::process(const cv::Mat& image_in) {
  const auto t0 = std::chrono::steady_clock::now();
  cv::Mat gray = image_in;
  if (gray.channels() == 3) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
  if (gray.type() != CV_8UC1 && gray.type() != CV_16UC1) {
    throw std::invalid_argument("BasaltFlowTracker: unsupported image type");
  }
  basalt::ManagedImage<uint16_t> img(gray.cols, gray.rows);
  for (int v = 0; v < gray.rows; ++v) {
    for (int u = 0; u < gray.cols; ++u) {
      img(u, v) = gray.depth() == CV_8U ? static_cast<uint16_t>(gray.at<uint8_t>(v, u) << 8) : gray.at<uint16_t>(v, u);
    }
  }
  impl_->old_pyramid = std::move(impl_->pyramid);
  impl_->pyramid = std::make_unique<Pyramid>();
  impl_->pyramid->setFromImage(img, config_.levels);

  stats_ = TrackerFrameStats{};
  dropped_.clear();
  std::map<int64_t, const TrackedFeature*> prev;
  for (const auto& f : features_) prev[f.id] = &f;

  // 1-2. Track with the forward-backward check.
  if (impl_->old_pyramid) {
    FlowStep step = impl_->flow->track(*impl_->old_pyramid, *impl_->pyramid, impl_->transforms);
    for (const auto& [id, why] : step.failed) {
      const DropReason r = why.first == TrackResult::kFailed ? DropReason::kKltFailed : DropReason::kForwardBackward;
      dropped_.push_back({*prev.at(id), r, why.second});
      (r == DropReason::kKltFailed ? stats_.lost_klt : stats_.rejected_fb)++;
      impl_->flow->erase(id);
    }
    if (config_.mask_valid_region) {
      for (auto it = step.tracked.begin(); it != step.tracked.end();) {
        if (!insideValid(it->second.translation().cast<double>())) {
          dropped_.push_back({*prev.at(it->first), DropReason::kLeftValidRegion});
          ++stats_.lost_klt;
          impl_->flow->erase(it->first);
          it = step.tracked.erase(it);
        } else {
          ++it;
        }
      }
    }
    impl_->transforms = std::move(step.tracked);
  }

  // 3. Detect in empty cells.
  std::vector<Eigen::Vector2d> current;
  for (const auto& [id, T] : impl_->transforms) current.push_back(T.translation().cast<double>());
  const auto accept = [&](const Eigen::Vector2d& c) { return !config_.mask_valid_region || insideValid(c); };
  const auto corners = detectKeypoints(impl_->pyramid->lvl(0), config_.grid_size, 1, current, accept);
  for (const Eigen::Vector2d& c : corners) {
    Eigen::AffineCompact2f T;
    T.setIdentity();
    T.translation() = c.cast<float>();
    impl_->transforms[next_id_] = T;
    impl_->flow->addPoint(*impl_->pyramid, next_id_, c.cast<float>());
    ++next_id_;
  }
  stats_.detected = static_cast<int>(corners.size());

  // Output, ordered by id like Basalt's observation map.
  std::vector<TrackedFeature> out;
  out.reserve(impl_->transforms.size());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [id, T] : impl_->transforms) {
    TrackedFeature f;
    f.id = id;
    f.uv = T.translation().cast<double>();
    if (!camera_->unproject(f.uv, f.bearing)) f.bearing = Eigen::Vector3d::Constant(nan);
    const auto it = prev.find(id);
    if (it != prev.end()) {
      f.age = it->second->age + 1;
      f.flow = f.uv - it->second->uv;
      ++stats_.tracked;
    }
    out.push_back(f);
  }
  std::vector<TrackedFeature> carried;
  for (const auto& f : out) {
    if (f.age > 0) carried.push_back(f);
  }
  updateMedianFlow(carried);
  features_.swap(out);
  stats_.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return features_;
}

}  // namespace msckf::vision
