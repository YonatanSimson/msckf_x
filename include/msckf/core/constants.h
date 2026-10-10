#pragma once

#include <Eigen/Core>

namespace msckf {

// Gravity in the global frame (z up), m/s^2. The accelerometer measures specific
// force a_m = R_GI^T (a - g) + b_a + n_a.
inline const Eigen::Vector3d kGravity(0.0, 0.0, -9.81);

}  // namespace msckf
