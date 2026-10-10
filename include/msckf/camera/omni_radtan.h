#pragma once

#include "msckf/camera/camera_model.h"

namespace msckf {

namespace internal {
struct RadtanDistortion;
}  // namespace internal

// Unified (omnidirectional) camera model of Mei and Rives (ICRA 2007) with
// radial-tangential distortion, as in Kalibr's omni-radtan. A point is first
// projected onto the unit sphere and then, from a center shifted by xi along
// the optical axis, onto a normalized plane:
//   m = (x, y) / (z + xi |p|),  uv = f * radtan(m) + c.
// Valid for z > -w |p| with w = xi for xi <= 1 and 1 / xi otherwise, which
// reaches past 90 degrees for any xi > 0.
// Parameters: fx fy cx cy xi k1 k2 p1 p2 (Kalibr: intrinsics [xi fu fv pu pv],
// distortion [k1 k2 r1 r2]).
class OmniRadtan final : public CameraModel {
 public:
  static constexpr int kNumParams = 9;

  OmniRadtan(int width, int height, const Eigen::VectorXd& params);

  std::string name() const override { return "omni-radtan"; }
  int numParams() const override { return kNumParams; }
  Eigen::VectorXd params() const override { return params_; }
  void setParams(const Eigen::VectorXd& params) override;

  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d& uv, Matrix23d* J_point = nullptr,
               Matrix2Xd* J_params = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d& bearing) const override;
  bool validPoint(const Eigen::Vector3d& p_c) const override;

  // Largest radius on the normalized plane m that the model accepts: the
  // radtan fold, and for xi > 1 the edge of the liftable disk 1 / sqrt(xi^2 - 1),
  // less a 1% margin.
  double maxRadius() const { return r_max_; }

 private:
  internal::RadtanDistortion distortion() const;
  void updateMaxRadius();

  Eigen::Matrix<double, kNumParams, 1> params_;
  double r_max_ = 0.0;
};

}  // namespace msckf
