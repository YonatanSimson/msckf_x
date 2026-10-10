#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

// Double Sphere model (Usenko, Demmel, Cremers, 3DV 2018). Closed-form in both
// directions, valid for fields of view beyond 180 degrees.
// Parameters: fx fy cx cy xi alpha.
class DoubleSphere final : public CameraModel {
 public:
  static constexpr int kNumParams = 6;

  DoubleSphere(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "double-sphere"; }
  int numParams() const override { return kNumParams; }
  Eigen::VectorXd params() const override { return params_; }
  void setParams(const Eigen::VectorXd& params) override;

  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point = nullptr,
               Matrix2Xd* J_params = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const override;
  bool validPoint(const Eigen::Vector3d& p_c) const override;

 private:
  Eigen::Matrix<double, kNumParams, 1> params_;
};

}  // namespace msckf
