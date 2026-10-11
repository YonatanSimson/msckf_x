#pragma once

#include "msckf/filter/filter_state.h"
#include "msckf/imu/preintegration.h"

#include <vector>

namespace msckf::filter {

// Discrete transition of the 15-dim IMU error over one propagation interval
// [t_k, t_k+1], and the covariance of the noise it accumulates:
//   e_k+1 = Phi e_k + w,  w ~ N(0, Q).
struct ImuTransition {
  Matrix15d Phi = Matrix15d::Identity();
  Matrix15d Q = Matrix15d::Zero();
};

// Closed-form transition from the preintegrated deltas (dR, dv, dp and their
// bias Jacobians, all in the IMU frame at t_k), with x_k at the linearization
// point lin_k and x_k+1 the propagated state:
//
//   dtheta' = (R_k+1^T R_k) dtheta + dR_dbg db_g
//   dv'     = dv - [v_k+1 - v_k - g T]x R_k dtheta + R_k (dV_dbg db_g + dV_dba db_a)
//   dp'     = dp + T dv - [p_k+1 - p_k - v_k T - g T^2 / 2]x R_k dtheta + R_k (dP_dbg db_g + dP_dba db_a)
//
// The bracketed terms are R_k dv and R_k dp written in states, so that with
// FEJ (lin_k = the estimate before the last update) the transition depends
// only on first estimates (Huang, Mourikis & Roumeliotis). Q maps the
// preintegration covariance into the global frame and adds the bias random
// walk over T (its coupling into the other states within the interval is
// neglected: second order in T).
ImuTransition imuTransition(const ImuPreintegration& pre, const NavState& lin_k, const NavState& x_k1,
                            const ImuNoise& noise);

// Propagates the IMU state of s to time t (IMU clock) with the samples (which
// must cover [s.t, t]): the nominal state by RK4 over each interval between
// samples with bias-corrected readings, P through imuTransition (the IMU
// block, and the cross terms with the rest of the state). Sets the IMU's FEJ
// copy to the result. Returns the bias-corrected angular velocity at t.
Eigen::Vector3d propagate(FilterState& s, const std::vector<ImuSample>& imu, double t, const ImuNoise& noise);

}  // namespace msckf::filter
