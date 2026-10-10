#include "msckf/sim/scenarios.h"

#include "msckf/sim/simulator.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace msckf {

namespace {

Sophus::SO3d fromRollPitchYaw(double roll, double pitch, double yaw) {
  return Sophus::SO3d::rotZ(yaw) * Sophus::SO3d::rotY(pitch) * Sophus::SO3d::rotX(roll);
}

int numControlPoses(double duration, double knot_dt) {
  return static_cast<int>(std::ceil(duration / knot_dt)) + 4;
}

SplineTrajectory makeHandheld(double duration, uint64_t seed) {
  constexpr double kKnotDt = 0.2;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> pos_jitter(0.0, 0.05), rot_jitter(0.0, 0.03);
  SplineTrajectory spline(0.0, kKnotDt);
  for (int k = 0; k < numControlPoses(duration, kKnotDt); ++k) {
    const double t = k * kKnotDt;
    const Eigen::Vector3d p(1.5 * std::sin(0.4 * t) + pos_jitter(rng),
                            1.0 * std::sin(0.55 * t + 1.0) + pos_jitter(rng),
                            1.2 + 0.3 * std::sin(0.7 * t) + pos_jitter(rng));
    const Sophus::SO3d R = fromRollPitchYaw(0.3 * std::sin(0.9 * t + 0.5) + rot_jitter(rng),
                                            0.25 * std::sin(0.7 * t) + rot_jitter(rng),
                                            1.2 * std::sin(0.25 * t) + rot_jitter(rng));
    spline.addControlPose(R, p);
  }
  return spline;
}

SplineTrajectory makeFastRotation(double duration, uint64_t seed) {
  constexpr double kKnotDt = 0.1;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> pos_jitter(0.0, 0.01);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);
  const double ph_r = phase(rng), ph_p = phase(rng), ph_y = phase(rng);
  SplineTrajectory spline(0.0, kKnotDt);
  for (int k = 0; k < numControlPoses(duration, kKnotDt); ++k) {
    const double t = k * kKnotDt;
    const Eigen::Vector3d p(0.3 * std::sin(0.8 * t) + pos_jitter(rng),
                            0.3 * std::sin(0.6 * t) + pos_jitter(rng),
                            1.2 + 0.1 * std::sin(1.0 * t) + pos_jitter(rng));
    // Peak yaw rate 2.5 * 2.0 = 5 rad/s.
    const Sophus::SO3d R = fromRollPitchYaw(0.6 * std::sin(3.1 * t + ph_r),
                                            0.8 * std::sin(2.6 * t + ph_p),
                                            2.5 * std::sin(2.0 * t + ph_y));
    spline.addControlPose(R, p);
  }
  return spline;
}

// Trapezoidal speed profile over a distance, starting and ending at rest.
struct SpeedProfile {
  double length, v_max, accel;
  double t_ramp = 0.0, t_total = 0.0;

  SpeedProfile(double L, double v, double a) : length(L), v_max(v), accel(a) {
    if (L < v * v / a) v_max = std::sqrt(L * a);  // triangular profile
    t_ramp = v_max / accel;
    t_total = 2.0 * t_ramp + (length - v_max * t_ramp) / v_max;
  }
  // Distance traveled and speed at time tau into the profile.
  void at(double tau, double& s, double& v) const {
    tau = std::clamp(tau, 0.0, t_total);
    const double t_cruise_end = t_total - t_ramp;
    if (tau < t_ramp) {
      v = accel * tau;
      s = 0.5 * accel * tau * tau;
    } else if (tau < t_cruise_end) {
      v = v_max;
      s = 0.5 * v_max * t_ramp + v_max * (tau - t_ramp);
    } else {
      const double r = t_total - tau;
      v = accel * r;
      s = length - 0.5 * accel * r * r;
    }
  }
};

// One leg of the cart path: a stop, a straight, or an arc of radius > 0.
struct CartLeg {
  double t_start = 0.0;
  double duration = 0.0;
  Eigen::Vector2d p0 = Eigen::Vector2d::Zero();
  double yaw0 = 0.0;
  double radius = 0.0;     // 0 for straights and stops
  double turn_sign = 0.0;  // +1 left, -1 right
  bool moving = false;
  SpeedProfile profile{1.0, 1.0, 1.0};
};

void sampleLeg(const CartLeg& leg, double t, Eigen::Vector2d& p, double& yaw, double& speed) {
  if (!leg.moving) {
    p = leg.p0;
    yaw = leg.yaw0;
    speed = 0.0;
    return;
  }
  double s;
  leg.profile.at(t - leg.t_start, s, speed);
  if (leg.radius == 0.0) {
    p = leg.p0 + s * Eigen::Vector2d(std::cos(leg.yaw0), std::sin(leg.yaw0));
    yaw = leg.yaw0;
    return;
  }
  const double sg = leg.turn_sign, R = leg.radius;
  const Eigen::Vector2d radial0(std::sin(leg.yaw0), -std::cos(leg.yaw0));
  const Eigen::Vector2d center = leg.p0 - sg * R * radial0;
  yaw = leg.yaw0 + sg * s / R;
  p = center + sg * R * Eigen::Vector2d(std::sin(yaw), -std::cos(yaw));
}

SplineTrajectory makeCart(double duration, uint64_t seed) {
  constexpr double kKnotDt = 0.2;
  constexpr double kHeight = 0.8;  // IMU height above ground
  constexpr double kStraightLength = 20.0, kStraightSpeed = 1.0, kStraightAccel = 0.5;
  constexpr double kTurnRadius = 2.0, kTurnSpeed = 0.5, kTurnAccel = 0.25;
  constexpr double kVibrationRad = 0.5 * M_PI / 180.0;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> stop_time(1.0, 4.0);

  // Lawnmower: stop, straight, stop, U-turn (alternating side), repeat.
  std::vector<CartLeg> legs;
  double t = 0.0, yaw = 0.0, turn = 1.0;
  Eigen::Vector2d p = Eigen::Vector2d::Zero();
  auto push = [&](CartLeg leg) {
    leg.t_start = t;
    leg.p0 = p;
    leg.yaw0 = yaw;
    if (leg.moving) leg.duration = leg.profile.t_total;
    t += leg.duration;
    double speed;
    sampleLeg(leg, t, p, yaw, speed);
    legs.push_back(leg);
  };
  const double t_needed = duration + 4.0 * kKnotDt;
  while (t < t_needed) {
    CartLeg stop;
    stop.duration = stop_time(rng);
    push(stop);
    CartLeg straight;
    straight.moving = true;
    straight.profile = SpeedProfile(kStraightLength, kStraightSpeed, kStraightAccel);
    push(straight);
    stop.duration = stop_time(rng);
    push(stop);
    CartLeg arc;
    arc.moving = true;
    arc.radius = kTurnRadius;
    arc.turn_sign = turn;
    arc.profile = SpeedProfile(M_PI * kTurnRadius, kTurnSpeed, kTurnAccel);
    push(arc);
    turn = -turn;
  }

  SplineTrajectory spline(0.0, kKnotDt);
  size_t leg = 0;
  for (int k = 0; k < numControlPoses(duration, kKnotDt); ++k) {
    const double tk = k * kKnotDt;
    while (leg + 1 < legs.size() && tk >= legs[leg + 1].t_start) ++leg;
    Eigen::Vector2d pk;
    double yk, speed;
    sampleLeg(legs[leg], tk, pk, yk, speed);
    // Ground vibration scales with speed, so stops are exactly stationary.
    const double vib = kVibrationRad * speed / kStraightSpeed;
    const Sophus::SO3d R = fromRollPitchYaw(vib * std::sin(2.0 * M_PI * 1.3 * tk),
                                            vib * std::sin(2.0 * M_PI * 0.9 * tk + 1.0), yk);
    const double bounce = 0.002 * (speed / kStraightSpeed) * std::sin(2.0 * M_PI * 1.7 * tk);
    spline.addControlPose(R, Eigen::Vector3d(pk.x(), pk.y(), kHeight + bounce));
  }
  return spline;
}

}  // namespace

std::string toString(Scenario scenario) {
  switch (scenario) {
    case Scenario::kHandheld: return "handheld";
    case Scenario::kCart: return "cart";
    case Scenario::kFastRotation: return "fast-rotation";
  }
  return "unknown";
}

SimConfig defaultSimConfig(Scenario scenario) {
  SimConfig config;
  if (scenario == Scenario::kCart) {
    config.ground_density = 1.0;
    config.wall_density = 1.0;
    config.ceiling_density = 0.0;
    config.wall_margin = 8.0;
    config.wall_height = 6.0;
  }
  return config;
}

SplineTrajectory makeScenario(Scenario scenario, double duration, uint64_t seed) {
  switch (scenario) {
    case Scenario::kHandheld: return makeHandheld(duration, seed);
    case Scenario::kCart: return makeCart(duration, seed);
    case Scenario::kFastRotation: return makeFastRotation(duration, seed);
  }
  throw std::invalid_argument("unknown scenario");
}

}  // namespace msckf
