#pragma once

#include "msckf/imu/imu_types.h"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace msckf::io {

/// EuRoC / TUM-VI IMU CSV (mav0/imu0/data.csv):
/// "t_ns, wx, wy, wz, ax, ay, az" with a '#' header. Times in seconds.
std::vector<ImuSample> readEurocImu(const std::string& path);

/// One row of EuRoC's state ground truth (state_groundtruth_estimate0/data.csv).
struct EurocState {
  double t = 0.0;                      ///< s
  Eigen::Vector3d v = Eigen::Vector3d::Zero();   ///< world velocity
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();  ///< gyro bias
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();  ///< accelerometer bias
};

/// EuRoC state ground truth: columns 8-10 velocity, 11-13 b_w, 14-16 b_a.
/// Empty for pose-only files (TUM-VI, LaMAria, KITTI conversions).
/// @throws std::runtime_error if the file cannot be opened.
std::vector<EurocState> readEurocStates(const std::string& path);

/// The front end's tracks of one image (tracker_eval tracks.csv).
struct TrackedFrame {
  double t = 0.0;  ///< camera time, s
  std::vector<std::pair<int64_t, Eigen::Vector2d>> uv;  ///< track id, raw pixel
};

/// tracker_eval's tracks.csv ("frame,t_ns,track,u,v" with a header), one entry
/// per frame index in order. @throws std::runtime_error if the file cannot be opened.
std::vector<TrackedFrame> readTracks(const std::string& path);

}  // namespace msckf::io
