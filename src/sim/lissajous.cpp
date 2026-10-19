#include "msckf/sim/lissajous.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace msckf {

namespace {

// A function of time with its first and second derivative: f, f', f''.
struct Jet {
  double v, d, dd;
};

Jet operator+(const Jet& a, const Jet& b) { return {a.v + b.v, a.d + b.d, a.dd + b.dd}; }
Jet operator-(const Jet& a) { return {-a.v, -a.d, -a.dd}; }
Jet operator*(const Jet& a, const Jet& b) {
  return {a.v * b.v, a.d * b.v + a.v * b.d, a.dd * b.v + 2.0 * a.d * b.d + a.v * b.dd};
}

Jet sqrt(const Jet& a) {
  const double r = std::sqrt(a.v), r1 = a.d / (2.0 * r);
  return {r, r1, (a.dd - 2.0 * r1 * r1) / (2.0 * r)};
}

Jet atan2(const Jet& y, const Jet& x) {
  const double den = x.v * x.v + y.v * y.v, num = x.v * y.d - y.v * x.d;
  const double num1 = x.v * y.dd - y.v * x.dd, den1 = 2.0 * (x.v * x.d + y.v * y.d);
  return {std::atan2(y.v, x.v), num / den, (num1 * den - num * den1) / (den * den)};
}

// amplitude sin(k t + phase).
Jet wave(double amplitude, double k, double phase, double t) {
  const double s = std::sin(k * t + phase), c = std::cos(k * t + phase);
  return {amplitude * s, amplitude * k * c, -amplitude * k * k * s};
}

Jet angle(const AngleWave& w, double omega, double t) {
  return Jet{w.offset, 0.0, 0.0} + wave(w.amplitude, w.multiple * omega, w.phase, t);
}

// A rotation with its body angular velocity and acceleration.
struct Spin {
  Sophus::SO3d R;
  Eigen::Vector3d omega = Eigen::Vector3d::Zero(), alpha = Eigen::Vector3d::Zero();
};

Spin axisRotation(const Eigen::Vector3d& axis, const Jet& a) {
  return {Sophus::SO3d::exp(a.v * axis), a.d * axis, a.dd * axis};
}

// R = R_a R_b: ω = R_bᵀ ω_a + ω_b; α = R_bᵀ α_a − ω_b × (R_bᵀ ω_a) + α_b.
Spin operator*(const Spin& a, const Spin& b) {
  const Eigen::Vector3d wa = b.R.inverse() * a.omega;
  return {a.R * b.R, wa + b.omega, b.R.inverse() * a.alpha - b.omega.cross(wa) + b.alpha};
}

// Velocity in F, per axis, with its two derivatives.
std::array<Jet, 3> pathVelocity(const LissajousConfig& c, double t) {
  std::array<Jet, 3> v;
  for (int i = 0; i < 3; ++i) {
    const double k = c.ratio[i] * c.omega;
    v[i] = wave(c.amplitude[i] * k, k, c.phase[i] + M_PI / 2.0, t);
  }
  return v;
}

Spin heading(const LissajousConfig& c, double t) {
  if (c.heading == Heading::kOff) return {};
  const auto [vx, vy, vz] = pathVelocity(c, t);
  const Spin yaw = axisRotation(Eigen::Vector3d::UnitZ(), atan2(vy, vx));
  if (c.heading == Heading::kPlanar) return yaw;
  return yaw * axisRotation(Eigen::Vector3d::UnitY(), -atan2(vz, sqrt(vx * vx + vy * vy)));
}

}  // namespace

LissajousTrajectory::LissajousTrajectory(const LissajousConfig& config, double duration)
    : config_(config), duration_(duration) {
  if (config.heading == Heading::kOff) return;
  for (double t = 0.0; t <= duration; t += 0.01) {
    const auto [vx, vy, vz] = pathVelocity(config, t);
    if (std::hypot(vx.v, vy.v) < 1e-3) throw std::invalid_argument("Lissajous: heading undefined, path stops");
  }
}

TrajectoryState LissajousTrajectory::evaluate(double t) const {
  const LissajousConfig& c = config_;
  Eigen::Vector3d p, v, a;
  for (int i = 0; i < 3; ++i) {
    const Jet x = wave(c.amplitude[i], c.ratio[i] * c.omega, c.phase[i], t);
    p[i] = x.v, v[i] = x.d, a[i] = x.dd;
  }
  const Spin body = heading(c, t) * axisRotation(Eigen::Vector3d::UnitZ(), angle(c.yaw, c.omega, t)) *
                    axisRotation(Eigen::Vector3d::UnitY(), angle(c.pitch, c.omega, t)) *
                    axisRotation(Eigen::Vector3d::UnitX(), angle(c.roll, c.omega, t));
  TrajectoryState x;
  x.R_GI = c.R_WF * body.R;
  x.p_GI = c.R_WF * p + c.p0;
  x.v_GI = c.R_WF * v;
  x.a_GI = c.R_WF * a;
  x.omega_I = body.omega;  // F is inertial: the figure attitude leaves body rates unchanged
  x.alpha_I = body.alpha;
  return x;
}

}  // namespace msckf
