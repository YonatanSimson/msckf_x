#include "msckf/camera/camera_model.h"

namespace msckf {

bool CameraModel::validPixel(const Eigen::Vector2d& uv) const {
  Eigen::Vector3d bearing;
  return inImage(uv) && unproject(uv, bearing);
}

bool CameraModel::visible(const Eigen::Vector3d& p_c) const {
  Eigen::Vector2d uv;
  return project(p_c, uv) && inImage(uv);
}

}  // namespace msckf
