#include "msckf/filter/observability.h"

#include "msckf/init/imu_init_state.h"

namespace msckf::filter {

namespace {

// d(alpha, beta, rho) / d(p_G) of an anchored point.
Eigen::Matrix3d anchoredJacobian(const AnchoredPoint& f) {
  const Eigen::Vector3d p_A = Eigen::Vector3d(f.x.x(), f.x.y(), 1.0) / f.x.z();
  const double z = p_A.z();
  Eigen::Matrix3d J;
  J << 1.0 / z, 0.0, -p_A.x() / (z * z), 0.0, 1.0 / z, -p_A.y() / (z * z), 0.0, 0.0, -1.0 / (z * z);
  return J * f.T_GA.so3().inverse().matrix();
}

}  // namespace

Eigen::MatrixXd unobservableDirections(const NavState& x, const std::vector<AnchoredPoint>& landmarks) {
  using namespace imu_err;
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(kDim + 3 * static_cast<long>(landmarks.size()), 4);
  N.block<3, 3>(kP, 0).setIdentity();
  N.block<3, 1>(kTheta, 3) = x.R_GI.inverse() * z;
  N.block<3, 1>(kP, 3) = z.cross(x.p_GI);
  N.block<3, 1>(kV, 3) = z.cross(x.v_GI);
  for (size_t l = 0; l < landmarks.size(); ++l) {
    const long r = kDim + 3 * static_cast<long>(l);
    const Eigen::Matrix3d J = anchoredJacobian(landmarks[l]);
    N.block<3, 3>(r, 0) = J;
    N.block<3, 1>(r, 3) = J * z.cross(landmarks[l].point());
  }
  return N;
}

}  // namespace msckf::filter
