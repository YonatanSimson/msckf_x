// Shared simulator test helper: analytic trajectory derivatives against
// central differences.

#pragma once

#include "msckf/sim/trajectory.h"

#include <algorithm>
#include <random>

namespace msckf {

// Largest relative error of each derivative over random times.
struct DerivativeErrors {
  double v = 0.0, a = 0.0, omega = 0.0, alpha = 0.0;
};

inline DerivativeErrors derivativeErrors(const Trajectory& traj) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> time(traj.minTime() + 0.01, traj.maxTime() - 0.01);
  constexpr double h = 1e-5;
  auto rel = [](const Eigen::Vector3d& x, const Eigen::Vector3d& fd) { return (x - fd).norm() / (1.0 + x.norm()); };
  DerivativeErrors e;
  for (int i = 0; i < 2000; ++i) {
    const double t = time(rng);
    const TrajectoryState s = traj.evaluate(t), sp = traj.evaluate(t + h), sm = traj.evaluate(t - h);
    e.v = std::max(e.v, rel(s.v_GI, (sp.p_GI - sm.p_GI) / (2 * h)));
    e.a = std::max(e.a, rel(s.a_GI, (sp.v_GI - sm.v_GI) / (2 * h)));
    // Body rate: R(t + h) = R(t - h) Exp(2 h w) to second order.
    e.omega = std::max(e.omega, rel(s.omega_I, (sm.R_GI.inverse() * sp.R_GI).log() / (2 * h)));
    e.alpha = std::max(e.alpha, rel(s.alpha_I, (sp.omega_I - sm.omega_I) / (2 * h)));
  }
  return e;
}

}  // namespace msckf
