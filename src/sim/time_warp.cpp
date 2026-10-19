#include "msckf/sim/time_warp.h"

#include <stdexcept>

namespace msckf {

namespace {

// Warped time w, rate w' and acceleration w'' at u = t − t0.
struct Warp {
  double w, rate, acc;
};

Warp warp(double u, double t_static, double t_ramp) {
  if (u <= t_static) return {0.0, 0.0, 0.0};
  if (u >= t_static + t_ramp) return {t_ramp / 2.0 + (u - t_static - t_ramp), 1.0, 0.0};
  const double tau = (u - t_static) / t_ramp, tau2 = tau * tau, tau3 = tau2 * tau;
  return {t_ramp * tau2 * tau2 * (tau2 - 3.0 * tau + 2.5), tau3 * (6.0 * tau2 - 15.0 * tau + 10.0),
          30.0 * tau2 * (tau - 1.0) * (tau - 1.0) / t_ramp};
}

}  // namespace

TimeWarpedTrajectory::TimeWarpedTrajectory(TrajectoryPtr base, double static_duration, double ramp_duration)
    : base_(std::move(base)), static_duration_(static_duration), ramp_duration_(ramp_duration) {
  if (static_duration < 0.0 || ramp_duration < 0.0) throw std::invalid_argument("time warp: negative duration");
  if (static_duration > 0.0 && ramp_duration <= 0.0) {
    throw std::invalid_argument("time warp: static start needs a ramp");
  }
  if (base_->maxTime() - base_->minTime() < ramp_duration / 2.0) {
    throw std::invalid_argument("time warp: base trajectory shorter than half the ramp");
  }
}

double TimeWarpedTrajectory::maxTime() const {
  return base_->maxTime() + static_duration_ + ramp_duration_ / 2.0;
}

TrajectoryState TimeWarpedTrajectory::evaluate(double t) const {
  const Warp s = warp(t - minTime(), static_duration_, ramp_duration_);
  TrajectoryState x = base_->evaluate(minTime() + s.w);
  // Chain rule: d/dt = w' d/ds; second derivatives pick up w'' times the first.
  x.a_GI = x.a_GI * s.rate * s.rate + x.v_GI * s.acc;
  x.v_GI *= s.rate;
  x.alpha_I = x.alpha_I * s.rate * s.rate + x.omega_I * s.acc;
  x.omega_I *= s.rate;
  return x;
}

}  // namespace msckf
