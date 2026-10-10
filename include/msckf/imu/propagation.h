#pragma once

#include "msckf/imu/imu_types.h"

namespace msckf {

// Bias-corrected angular velocity and specific force at one instant.
struct ImuReading {
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
};

// One classical RK4 step of the nominal kinematics over [t, t + dt]:
//   q_dot = 1/2 q (x) [omega, 0],  v_dot = R_GI acc + g,  p_dot = v.
// r0, r_mid, r1 are the readings at t, t + dt/2 and t + dt. The quaternion is
// renormalized at the end of the step.
NavState rk4Step(const NavState& x, const ImuReading& r0, const ImuReading& r_mid,
                 const ImuReading& r1, double dt);

// Linear interpolation between two readings; how the filter gets the midpoint
// reading when only discrete samples exist.
ImuReading interpolate(const ImuReading& a, const ImuReading& b, double alpha);

}  // namespace msckf
