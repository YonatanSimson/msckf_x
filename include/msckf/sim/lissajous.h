#pragma once

#include "msckf/sim/trajectory.h"

namespace msckf {

/// How the body's heading follows the path (user's simulator plan, 2b).
enum class Heading {
  kOff,      ///< no path following: attitude is the offset alone
  kPlanar,   ///< yaw along the horizontal velocity in F
  kTangent,  ///< yaw and pitch along the full velocity in F
};

/// One attitude angle: offset + amplitude sin(multiple Ω t + phase) (rad).
struct AngleWave {
  double offset = 0.0;
  double amplitude = 0.0;
  double multiple = 1.0;  ///< of the path frequency Ω
  double phase = 0.0;
};

/// A 3D Lissajous path in a figure frame F, placed in the world by a constant
/// R_WF and p0. The default is the Gerono figure-8 with a vertical wobble
/// (ratios 1:2:3), a camera looking sideways (yaw offset 90°) and roll and
/// pitch sway.
struct LissajousConfig {
  Eigen::Vector3d amplitude{3.0, 1.5, 0.3};  ///< per axis of F (m)
  Eigen::Vector3d ratio{1.0, 2.0, 3.0};      ///< frequency multiples a:b:c of Ω
  Eigen::Vector3d phase = Eigen::Vector3d::Zero();  ///< per axis (rad)
  double omega = 2.0 * M_PI / 20.0;          ///< path frequency Ω (rad/s): one figure-8 in 20 s
  Heading heading = Heading::kPlanar;
  AngleWave roll{0.0, 8.0 * M_PI / 180.0, 5.0, 0.0};   ///< about x, applied first
  AngleWave pitch{0.0, 8.0 * M_PI / 180.0, 7.0, 0.0};  ///< about y
  AngleWave yaw{M_PI / 2.0, 0.0, 1.0, 0.0};            ///< about z, applied last
  Sophus::SO3d R_WF;                         ///< figure attitude in the world
  Eigen::Vector3d p0{0.0, 0.0, 1.5};         ///< figure centre in the world (m)
};

/// R_FI = R_head R_off with R_head = Rz(ψ) Ry(−θ) along the path and
/// R_off = Rz(yaw) Ry(pitch) Rx(roll); R_GI = R_WF R_FI, p = R_WF p_F + p0.
/// Every derivative is analytic (forward-mode second-order jets).
class LissajousTrajectory : public Trajectory {
 public:
  /// Defined on [0, duration].
  /// @throws std::invalid_argument if the heading follows the path and the
  ///         path's horizontal speed in F comes near zero (yaw undefined).
  LissajousTrajectory(const LissajousConfig& config, double duration);

  TrajectoryState evaluate(double t) const override;
  double minTime() const override { return 0.0; }
  double maxTime() const override { return duration_; }

 private:
  LissajousConfig config_;
  double duration_;
};

}  // namespace msckf
