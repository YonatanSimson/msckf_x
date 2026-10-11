#include "msckf/filter/msckf.h"

#include "msckf/filter/propagator.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace msckf::filter {

Msckf::Msckf(CameraModelPtr camera, const ImuInitState& init, const Sophus::SO3d& R_IC, const Eigen::Vector3d& p_IC,
             double t_d, const Eigen::Matrix<double, 7, 7>& calib_cov, const MsckfOptions& options)
    : camera_(std::move(camera)), options_(options), state_(init, R_IC, p_IC, t_d, options.state, calib_cov) {}

void Msckf::addImu(const ImuSample& sample) { imu_.push_back(sample); }

FrameResult Msckf::addImage(double t_cam, const std::vector<FeatureObservation>& observations) {
  const auto t0 = std::chrono::steady_clock::now();
  FrameResult result;

  // 1. Propagate and clone.
  const Eigen::Vector3d omega = propagate(state_, imu_, t_cam + state_.t_d, options_.noise);
  state_.addClone(t_cam, omega);

  // 2. Extend the tracks.
  for (const FeatureObservation& o : observations) {
    Eigen::Vector3d bearing;
    if (!camera_->unproject(o.uv, bearing)) continue;
    FeatureTrack& tr = tracks_[o.id];
    tr.id = o.id;
    tr.t_cam.push_back(t_cam);
    tr.uv.push_back(o.uv);
    tr.bearing.push_back(bearing);
  }

  // 3. Tracks that ended, and those in the oldest clone if it is about to go.
  const bool full = static_cast<int>(state_.clones().size()) > options_.max_clones;
  const double oldest = state_.clones().front().t_cam;
  std::vector<FeatureTrack> use;
  std::vector<int64_t> done;
  for (auto& [id, tr] : tracks_) {
    const bool ended = tr.t_cam.back() != t_cam;
    const bool in_oldest = full && tr.t_cam.front() == oldest;
    if (ended || (in_oldest && static_cast<int>(tr.t_cam.size()) >= options_.update.min_observations)) {
      use.push_back(std::move(tr));
      done.push_back(id);
    } else if (in_oldest) {
      // Too short to use: forget its observation in the clone that goes.
      tr.t_cam.erase(tr.t_cam.begin());
      tr.uv.erase(tr.uv.begin());
      tr.bearing.erase(tr.bearing.begin());
      if (tr.t_cam.empty()) done.push_back(id);
    }
  }
  for (int64_t id : done) tracks_.erase(id);
  result.update = featureUpdate(state_, *camera_, use, options_.update);

  // 4. Marginalize the oldest clone.
  if (full) state_.marginalizeClone(0);

  // Keep the IMU samples still needed (from the one before the state time).
  const auto later = [](double v, const ImuSample& x) { return v < x.t; };
  auto keep = std::upper_bound(imu_.begin(), imu_.end(), state_.t, later);
  if (keep - imu_.begin() > 1) imu_.erase(imu_.begin(), keep - 1);

  result.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

}  // namespace msckf::filter
