#pragma once

#include <sophus/se3.hpp>

#include <string>
#include <vector>

namespace msckf::eval {

enum class AlignmentType {
  kNone,
  kPosYaw,  // 4-DOF: translation + yaw about gravity, the VIO gauge
  kSE3,     // 6-DOF
  kSim3,    // 7-DOF, includes scale (monocular without IMU, or scale error)
};

AlignmentType parseAlignment(const std::string& name);
std::string toString(AlignmentType type);

// Maps estimate positions into the ground-truth frame: p_gt ~ s R p_est + t.
struct Alignment {
  Sophus::SO3d R;
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  double scale = 1.0;

  Eigen::Vector3d apply(const Eigen::Vector3d& p) const { return scale * (R * p) + t; }
  Sophus::SE3d apply(const Sophus::SE3d& T) const {
    return Sophus::SE3d(R * T.so3(), apply(T.translation()));
  }
};

// Least-squares alignment of estimate positions to ground-truth positions.
// PosYaw is the closed-form 4-DOF solution; SE3 and Sim3 are Umeyama (1991).
Alignment align(const std::vector<Eigen::Vector3d>& est, const std::vector<Eigen::Vector3d>& gt,
                AlignmentType type);

}  // namespace msckf::eval
