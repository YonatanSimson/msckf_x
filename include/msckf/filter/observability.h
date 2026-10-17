#pragma once

#include "msckf/geometry/triangulation.h"
#include "msckf/imu/imu_types.h"

#include <Eigen/Core>

#include <vector>

namespace msckf::filter {

// The 4 directions a visual-inertial system cannot observe (global position
// and yaw about gravity), in this filter's error state at the IMU state x:
// rows [dtheta, dp, dv, db_g, db_a] (body-frame dtheta, plan conventions), then
// d(alpha, beta, rho) per landmark; columns translation x, y, z, then yaw:
//   translation: dp = e_i, every landmark moved by e_i;
//   yaw (z up):  dtheta = R^T z, dp = z x p, dv = z x v, a landmark moved by z x p_f.
// With FEJ, Phi(k+1, k) N(x_k) = N(x_k+1) and H N = 0 at the linearization points.
Eigen::MatrixXd unobservableDirections(const NavState& x, const std::vector<AnchoredPoint>& landmarks);

}  // namespace msckf::filter
