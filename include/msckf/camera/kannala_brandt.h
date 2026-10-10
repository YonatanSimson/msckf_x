#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

// Kannala-Brandt equidistant model with four distortion coefficients (KB4):
//   d(theta) = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
// where theta is the angle from the optical axis. Handles theta > 90 degrees.
// Parameters: fx fy cx cy k1 k2 k3 k4.
class KannalaBrandt final : public CameraModel {
 public:
  static constexpr int kNumParams = 8;

  KannalaBrandt(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "kannala-brandt"; }
  int numParams() const override { return kNumParams; }
  Eigen::VectorXd params() const override { return params_; }
  void setParams(const Eigen::VectorXd& params) override;

  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point = nullptr,
               Matrix2Xd* J_params = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const override;
  bool validPoint(const Eigen::Vector3d& p_c) const override;

  // Largest angle for which d(theta) stays monotonic.
  double maxTheta() const { return theta_max_; }

 private:
  double poly(double theta, double* d_dtheta = nullptr) const;
  void updateMaxTheta();

  Eigen::Matrix<double, kNumParams, 1> params_;
  double theta_max_ = 0.0;
};

}  // namespace msckf
