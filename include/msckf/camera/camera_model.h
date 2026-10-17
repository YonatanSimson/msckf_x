#pragma once

#include <Eigen/Core>

#include <memory>
#include <string>

namespace msckf {

using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Matrix2Xd = Eigen::Matrix<double, 2, Eigen::Dynamic>;

// Common interface for every camera model. All operations are in double
// precision and work on raw (distorted) pixels and unit bearing vectors; nothing
// downstream assumes a pinhole, so lenses beyond 180 degrees need no special case.
class CameraModel {
 public:
  CameraModel(int width, int height) : width_(width), height_(height) {}
  virtual ~CameraModel() = default;

  virtual std::string name() const = 0;
  virtual int numParams() const = 0;
  // Parameter order is model-specific and always starts with fx fy cx cy.
  virtual Eigen::VectorXd params() const = 0;
  virtual void setParams(const Eigen::VectorXd& params) = 0;

  // Projects a point in the camera frame to a pixel. Returns false when the
  // point is outside the model's validity region; uv is then unspecified.
  // J_point is d(uv)/d(p_c) (2x3), J_params is d(uv)/d(params) (2 x numParams).
  virtual bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv,
                       Matrix23d* J_point = nullptr,
                       Matrix2Xd* J_params = nullptr) const = 0;

  // Unprojects a pixel to a unit bearing (never a point on z = 1, so rays past
  // 90 degrees stay representable). Returns false outside the valid region.
  virtual bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const = 0;

  // The model's own validity region for 3D points (ignores image bounds).
  virtual bool validPoint(const Eigen::Vector3d& p_c) const = 0;

  // True if uv is inside the image and inside the model's unprojection domain.
  bool validPixel(const Eigen::Vector2d& uv) const;

  // True if p_c is valid for the model and projects inside the image.
  bool visible(const Eigen::Vector3d& p_c) const;

  int width() const { return width_; }
  int height() const { return height_; }
  bool inImage(const Eigen::Vector2d& uv) const {
    return uv.x() >= 0.0 && uv.y() >= 0.0 && uv.x() <= width_ - 1.0 && uv.y() <= height_ - 1.0;
  }

 protected:
  int width_;
  int height_;
};

using CameraModelPtr = std::shared_ptr<CameraModel>;

/// Pixels per radian of bearing at the image centre (1 px step along u): the
/// focal length of a pinhole, its equivalent for wide-angle models.
/// @throws std::runtime_error if the centre does not unproject.
double pixelsPerRadian(const CameraModel& camera);

}  // namespace msckf
