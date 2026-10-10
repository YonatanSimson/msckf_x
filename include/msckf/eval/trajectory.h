#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <string>
#include <vector>

namespace msckf::eval {

struct StampedPose {
  double t = 0.0;
  Sophus::SE3d T_GI;  // IMU (body) pose in the global frame
};

using Trajectory = std::vector<StampedPose>;

// TUM format, one pose per line: "t tx ty tz qx qy qz qw"; '#' starts a comment.
Trajectory readTum(const std::string& path);
void writeTum(const std::string& path, const Trajectory& trajectory);

// EuRoC ground-truth CSV: "t_ns, px, py, pz, qw, qx, qy, qz, ..." with a header.
Trajectory readEurocGroundTruth(const std::string& path);

// Reads either format: "tum" or "euroc".
Trajectory readTrajectory(const std::string& path, const std::string& format);

// Pose at time t, interpolated between the bracketing samples (linear position,
// geodesic rotation). Returns false if t is outside the trajectory or the
// bracketing samples are more than max_gap apart. Requires sorted timestamps.
bool interpolatePose(const Trajectory& trajectory, double t, double max_gap, Sophus::SE3d& T);

// Estimate poses paired with ground truth interpolated at the same times.
struct Association {
  std::vector<double> t;
  std::vector<Sophus::SE3d> est;
  std::vector<Sophus::SE3d> gt;
  std::vector<size_t> est_index;  // index into the estimate trajectory
  size_t size() const { return t.size(); }
};

Association associate(const Trajectory& estimate, const Trajectory& ground_truth,
                      double max_gap = 0.05);

}  // namespace msckf::eval
