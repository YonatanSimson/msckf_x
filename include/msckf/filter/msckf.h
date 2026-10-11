#pragma once

#include "msckf/camera/camera_model.h"
#include "msckf/filter/feature_update.h"
#include "msckf/filter/filter_state.h"

#include <map>
#include <optional>
#include <vector>

namespace msckf::filter {

struct MsckfOptions {
  StateOptions state;
  ImuNoise noise;
  UpdateOptions update;
  int max_clones = 11;  // sliding window
};

// One feature observation in an image: track id and raw pixel.
struct FeatureObservation {
  int64_t id = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
};

struct FrameResult {
  UpdateStats update;
  double time_ms = 0.0;
};

// Monocular MSCKF (plan, Stage 3). Per image:
//  1. propagate to the image time t_cam + t_d and clone the IMU pose;
//  2. append the observations to their tracks;
//  3. update with every track that ended (not seen in this image) and, when
//     the window is over max_clones, with every track seen in the oldest
//     clone; tracks used are discarded;
//  4. marginalize the oldest clone when the window is over max_clones.
class Msckf {
 public:
  // calib_cov: covariance of [dtheta_IC, dp_IC, dt_d] (rows of what is not
  // estimated are ignored).
  Msckf(CameraModelPtr camera, const ImuInitState& init, const Sophus::SO3d& R_IC, const Eigen::Vector3d& p_IC,
        double t_d, const Eigen::Matrix<double, 7, 7>& calib_cov, const MsckfOptions& options);

  // IMU samples in time order; they must cover each image time before the
  // image is added.
  void addImu(const ImuSample& sample);
  FrameResult addImage(double t_cam, const std::vector<FeatureObservation>& observations);

  const FilterState& state() const { return state_; }
  const MsckfOptions& options() const { return options_; }

 private:
  CameraModelPtr camera_;
  MsckfOptions options_;
  FilterState state_;
  std::vector<ImuSample> imu_;
  std::map<int64_t, FeatureTrack> tracks_;
};

}  // namespace msckf::filter
