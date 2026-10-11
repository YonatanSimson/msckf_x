#include "msckf/filter/filter_state.h"

#include <vector>

namespace msckf::filter {

FilterState::FilterState(const ImuInitState& init, const Sophus::SO3d& R_IC_in, const Eigen::Vector3d& p_IC_in,
                         double t_d_in, const StateOptions& options, const Eigen::Matrix<double, 7, 7>& calib_cov)
    : t(init.t),
      nav(init.nav),
      nav_fej(init.nav),
      gyro_bias(init.gyro_bias),
      acc_bias(init.acc_bias),
      R_IC(R_IC_in),
      p_IC(p_IC_in),
      t_d(t_d_in),
      options_(options) {
  // Calibration rows of calib_cov kept: extrinsic [0, 6), time offset 6.
  std::vector<int> keep;
  if (options_.estimate_extrinsic) {
    for (int i = 0; i < 6; ++i) keep.push_back(i);
  }
  if (options_.estimate_time_offset) keep.push_back(6);
  clones_start_ = imu_err::kDim + static_cast<int>(keep.size());
  P = Eigen::MatrixXd::Zero(clones_start_, clones_start_);
  P.topLeftCorner<imu_err::kDim, imu_err::kDim>() = init.P;
  for (size_t i = 0; i < keep.size(); ++i) {
    for (size_t j = 0; j < keep.size(); ++j) P(imu_err::kDim + i, imu_err::kDim + j) = calib_cov(keep[i], keep[j]);
  }
}

int FilterState::timeOffsetIndex() const {
  if (!options_.estimate_time_offset) return -1;
  return imu_err::kDim + (options_.estimate_extrinsic ? 6 : 0);
}

void FilterState::inject(const Eigen::VectorXd& dx) {
  using namespace imu_err;
  nav.R_GI = nav.R_GI * Sophus::SO3d::exp(dx.segment<3>(kTheta));
  nav.p_GI += dx.segment<3>(kP);
  nav.v_GI += dx.segment<3>(kV);
  gyro_bias += dx.segment<3>(kBg);
  acc_bias += dx.segment<3>(kBa);
  if (const int e = extrinsicIndex(); e >= 0) {
    R_IC = R_IC * Sophus::SO3d::exp(dx.segment<3>(e));
    p_IC += dx.segment<3>(e + 3);
  }
  if (const int d = timeOffsetIndex(); d >= 0) t_d += dx(d);
  for (size_t i = 0; i < clones_.size(); ++i) {
    const int c = cloneIndex(i);
    clones_[i].R_GI = clones_[i].R_GI * Sophus::SO3d::exp(dx.segment<3>(c));
    clones_[i].p_GI += dx.segment<3>(c + 3);
  }
}

void FilterState::addClone(double t_cam, const Eigen::Vector3d& omega) {
  using namespace imu_err;
  const int n = dim();
  // Clone error = J * state error: [dtheta + omega dt_d; dp + v dt_d].
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n);
  J.block<3, 3>(0, kTheta).setIdentity();
  J.block<3, 3>(3, kP).setIdentity();
  if (const int d = timeOffsetIndex(); d >= 0) {
    J.block<3, 1>(0, d) = omega;
    J.block<3, 1>(3, d) = nav.v_GI;
  }
  const Eigen::MatrixXd PJt = P * J.transpose();
  Eigen::MatrixXd P_new(n + 6, n + 6);
  P_new.topLeftCorner(n, n) = P;
  P_new.topRightCorner(n, 6) = PJt;
  P_new.bottomLeftCorner(6, n) = PJt.transpose();
  P_new.bottomRightCorner(6, 6) = J * PJt;
  P = std::move(P_new);
  clones_.push_back({t_cam, nav.R_GI, nav.p_GI, nav.R_GI, nav.p_GI});
}

void FilterState::marginalizeClone(size_t i) {
  const int c = cloneIndex(i);
  const int n = dim();
  const int tail = n - c - 6;
  Eigen::MatrixXd P_new(n - 6, n - 6);
  P_new.topLeftCorner(c, c) = P.topLeftCorner(c, c);
  P_new.topRightCorner(c, tail) = P.topRightCorner(c, tail);
  P_new.bottomLeftCorner(tail, c) = P.bottomLeftCorner(tail, c);
  P_new.bottomRightCorner(tail, tail) = P.bottomRightCorner(tail, tail);
  P = std::move(P_new);
  clones_.erase(clones_.begin() + static_cast<long>(i));
}

}  // namespace msckf::filter
