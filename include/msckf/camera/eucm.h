#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

// Extended Unified Camera Model (Khomutenko et al., RA-L 2016), with the
// validity region as given in the Double Sphere paper.
// Parameters: fx fy cx cy alpha beta.
class Eucm final : public CameraModel {
 public:
  static constexpr int kNumParams = 6;

  Eucm(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "eucm"; }
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
