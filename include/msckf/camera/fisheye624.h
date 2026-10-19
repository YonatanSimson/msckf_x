#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

/// Project Aria's Fisheye624 model (also called FisheyeRadTanThinPrism; the
/// SLAM cameras of the Aria glasses, LaMAria): an equidistant fisheye with six
/// radial coefficients, followed by tangential and thin-prism distortion of the
/// radially distorted coordinates. With theta the angle from the optical axis
/// and (a, b) the direction of the point in the image plane,
///   d(theta) = theta (1 + k0 theta^2 + k1 theta^4 + ... + k5 theta^12),
///   w = d(theta) (a, b) / |(a, b)|,
///   w_d = w + 2 (w . p) w + |w|^2 p + (s0 |w|^2 + s1 |w|^4, s2 |w|^2 + s3 |w|^4),
///   uv = (fx, fy) * w_d + (cx, cy).
/// The model is defined for points in front of the camera (theta < 90 degrees)
/// and up to the angle where d stays monotonic.
/// Parameters: fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3.
class Fisheye624 final : public CameraModel {
 public:
  static constexpr int kNumParams = 16;

  Fisheye624(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "fisheye624"; }
  int numParams() const override { return kNumParams; }
  Eigen::VectorXd params() const override { return params_; }
  void setParams(const Eigen::VectorXd& params) override;

  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point = nullptr,
               Matrix2Xd* J_params = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const override;
  bool validPoint(const Eigen::Vector3d& p_c) const override;

  /// Largest angle from the optical axis for which d(theta) stays monotonic.
  double maxTheta() const { return theta_max_; }

 private:
  /// d(theta) and optionally its derivative.
  double radial(double theta, double* d_dtheta = nullptr) const;
  /// Tangential and thin-prism distortion of the radially distorted w, with d(w_d)/dw.
  Eigen::Vector2d distort(const Eigen::Vector2d& w, Eigen::Matrix2d* J_w = nullptr) const;
  /// Inverts distort by Newton; false if it does not converge.
  bool undistort(const Eigen::Vector2d& w_d, Eigen::Vector2d& w) const;
  /// Solves d(theta) = rd on [0, theta_max] by Newton; false if it does not converge.
  bool solveTheta(double rd, double& theta) const;
  /// d(uv)/d(params) from the intermediate quantities of project.
  void paramJacobian(const Eigen::Vector2d& w, const Eigen::Vector2d& w_d, const Eigen::Vector2d& dw_dk,
                     const Eigen::Matrix2d& J_w, double theta2, Matrix2Xd& J) const;
  /// Recomputes theta_max_ after the parameters change.
  void updateMaxTheta();

  Eigen::Matrix<double, kNumParams, 1> params_;
  double theta_max_ = 0.0;
};

}  // namespace msckf
