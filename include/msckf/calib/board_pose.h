#pragma once

#include "msckf/calib/aprilgrid.h"
#include "msckf/camera/camera_model.h"

#include <sophus/se3.hpp>

#include <vector>

namespace msckf::calib {

struct BoardPoseResult {
  Sophus::SE3d T_cam_board;
  // observed - projected pixel, per observed corner (same order as the
  // observation). Only meaningful when ok.
  std::vector<Eigen::Vector2d> residuals;
  double rms = 0.0;  // sqrt(mean |r|^2), px
  int iterations = 0;
  bool ok = false;
};

// Fits the board pose to one image's corners with the intrinsics held fixed:
// a planar PnP on the unprojected bearings for the start, then Levenberg-
// Marquardt on pixel residuals through the camera model.
BoardPoseResult fitBoardPose(const CameraModel& camera, const AprilGrid& grid,
                             const GridObservation& obs, int max_iterations = 30);

}  // namespace msckf::calib
