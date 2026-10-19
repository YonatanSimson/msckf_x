#pragma once

#include "msckf/sim/trajectory.h"

namespace msckf {

/// A trajectory driven by warped time s(t), for a start from rest (user's
/// simulator plan, 2d). The body holds the base trajectory's first pose for
/// static_duration, then s accelerates over ramp_duration with a smootherstep
/// rate (s' = 6τ⁵ − 15τ⁴ + 10τ³, so s'' and the jerk vanish at both ends: no
/// steps in the accelerometer), then follows the base at full speed.
class TimeWarpedTrajectory : public Trajectory {
 public:
  /// static_duration = ramp_duration = 0 leaves the base unchanged.
  /// @throws std::invalid_argument for negative durations, a static start
  ///         without a ramp, or a base shorter than half the ramp.
  TimeWarpedTrajectory(TrajectoryPtr base, double static_duration, double ramp_duration);

  TrajectoryState evaluate(double t) const override;
  double minTime() const override { return base_->minTime(); }
  double maxTime() const override;

 private:
  TrajectoryPtr base_;
  double static_duration_;
  double ramp_duration_;
};

}  // namespace msckf
