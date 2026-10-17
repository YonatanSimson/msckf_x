#include "msckf/camera/camera_model.h"

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>

namespace msckf {

bool CameraModel::validPixel(const Eigen::Vector2d& uv) const {
  Eigen::Vector3d bearing;
  return inImage(uv) && unproject(uv, bearing);
}

bool CameraModel::visible(const Eigen::Vector3d& p_c) const {
  Eigen::Vector2d uv;
  return project(p_c, uv) && inImage(uv);
}

double pixelsPerRadian(const CameraModel& camera) {
  const Eigen::Vector2d c(0.5 * camera.width(), 0.5 * camera.height());
  Eigen::Vector3d b0, b1;
  if (!camera.unproject(c, b0) || !camera.unproject(c + Eigen::Vector2d(1.0, 0.0), b1)) {
    throw std::runtime_error("camera centre does not unproject");
  }
  return 1.0 / std::atan2(b0.cross(b1).norm(), b0.dot(b1));
}

}  // namespace msckf
