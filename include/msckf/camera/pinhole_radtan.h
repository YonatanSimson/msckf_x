#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

namespace internal {
struct RadtanDistortion;
}  // namespace internal

// Pinhole with Brown-Conrady (radial-tangential) distortion.
// Parameters: fx fy cx cy k1 k2 p1 p2 k3.
class PinholeRadtan final : public CameraModel {
 public:
  static constexpr int kNumParams = 9;

  PinholeRadtan(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "pinhole-radtan"; }
  int numParams() const override { return kNumParams; }
  Eigen::VectorXd params() const override { return params_; }
  void setParams(const Eigen::VectorXd& params) override;

  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point = nullptr,
               Matrix2Xd* J_params = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const override;
  bool validPoint(const Eigen::Vector3d& p_c) const override;

  // Largest normalized radius for which the radial distortion stays monotonic.
  double maxRadius() const { return r_max_; }

 private:
  internal::RadtanDistortion distortion() const;
  void updateMaxRadius();

  Eigen::Matrix<double, kNumParams, 1> params_;
  double r_max_ = 0.0;
};

}  // namespace msckf
