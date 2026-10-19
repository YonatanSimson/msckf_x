#include "msckf/sim/scenarios.h"

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/sim/lissajous.h"
#include "msckf/sim/simulator.h"
#include "msckf/sim/spline.h"
#include "msckf/sim/time_warp.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace msckf {

namespace {

constexpr double kFigure8Static = 2.0;  // s at rest before the figure-8
constexpr double kFigure8Ramp = 2.0;    // s to reach full speed

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
  // Push surges (a person pushing the cart): speed varies by push_speed *
  // (v / v_max) at push_hz, so stops stay exactly stationary.
  double push_speed = 0.0, push_hz = 0.0, push_phase = 0.0;
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
  if (leg.push_speed > 0.0) {
    // s += A sin(w tau + phi) / w with A = push_speed * v / v_max, so the speed
    // gains about A cos(w tau + phi) (the slow change of A is neglected).
    const double w = 2.0 * M_PI * leg.push_hz, tau = t - leg.t_start;
    const double amp = leg.push_speed * speed / leg.profile.v_max;
    s += amp * (std::sin(w * tau + leg.push_phase) - std::sin(leg.push_phase)) / w;
    speed += amp * std::cos(w * tau + leg.push_phase);
  }
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
  // Push surges: +-0.15 m/s at 0.6 Hz at cruise speed (peak 0.57 m/s^2).
  constexpr double kPushSpeed = 0.15, kPushHz = 0.6;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> stop_time(1.0, 4.0), phase(0.0, 2.0 * M_PI);

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
    straight.push_speed = kPushSpeed;
    straight.push_hz = kPushHz;
    straight.push_phase = phase(rng);
    push(straight);
    stop.duration = stop_time(rng);
    push(stop);
    CartLeg arc;
    arc.moving = true;
    arc.radius = kTurnRadius;
    arc.turn_sign = turn;
    arc.profile = SpeedProfile(M_PI * kTurnRadius, kTurnSpeed, kTurnAccel);
    arc.push_speed = kPushSpeed;
    arc.push_hz = kPushHz;
    arc.push_phase = phase(rng);
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

// KITTI-like driving, integrated at 100 Hz: the car starts rolling at cruise
// speed; straights of 60-150 m at a cruise speed of 8-12 m/s (re-drawn
// halfway along, so speed also changes on straights), braking at 1.5 m/s^2 to
// 5 m/s for a 90 degree turn of 15 m radius (left or right at random), then
// accelerating at 1 m/s^2. Body pitch and roll follow the longitudinal and
// lateral acceleration (0.01 rad per m/s^2), plus road vibration.
SplineTrajectory makeCar(double duration, uint64_t seed) {
  constexpr double kKnotDt = 0.1, kDt = 0.01;
  constexpr double kHeight = 1.7;  // IMU height above the road (KITTI's OXTS is on the roof)
  constexpr double kTurnRadius = 15.0, kTurnSpeed = 5.0, kBrake = 1.5, kAccel = 1.0;
  constexpr double kTilt = 0.01, kVibrationRad = 0.2 * M_PI / 180.0;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> cruise(8.0, 12.0), straight(60.0, 150.0), unit(0.0, 1.0);

  double x = 0.0, y = 0.0, yaw = 0.0, v = cruise(rng), target = cruise(rng);
  double left = straight(rng), half = 0.5 * left, turned = 0.0, turn_sign = 0.0;
  bool turning = false;
  double a_long = 0.0, a_lat = 0.0;
  std::vector<std::pair<Sophus::SO3d, Eigen::Vector3d>> poses;
  const int n_steps = static_cast<int>(std::ceil((duration + 6.0 * kKnotDt) / kDt));
  const int knot_every = static_cast<int>(std::lround(kKnotDt / kDt));
  for (int i = 0; i <= n_steps; ++i) {
    const double t = i * kDt;
    if (i % knot_every == 0) {
      const double vib = kVibrationRad * v / 10.0;
      const Sophus::SO3d R = fromRollPitchYaw(kTilt * a_lat + vib * std::sin(2.0 * M_PI * 1.7 * t),
                                              -kTilt * a_long + vib * std::sin(2.0 * M_PI * 2.3 * t + 1.0), yaw);
      poses.emplace_back(R, Eigen::Vector3d(x, y, kHeight));
    }
    double omega = 0.0;
    if (turning) {
      omega = turn_sign * v / kTurnRadius;
      turned += std::abs(omega) * kDt;
      if (turned >= 0.5 * M_PI) {
        turning = false;
        left = straight(rng);
        half = 0.5 * left;
        target = cruise(rng);
      }
    } else {
      // Brake in time for the turn; re-draw the cruise speed halfway.
      if (left <= (v * v - kTurnSpeed * kTurnSpeed) / (2.0 * kBrake) + 1.0) target = kTurnSpeed;
      if (left < half && half > 0.0) {
        target = target == kTurnSpeed ? kTurnSpeed : cruise(rng);
        half = -1.0;
      }
      if (left <= 0.0) {
        turning = true;
        turned = 0.0;
        turn_sign = unit(rng) < 0.5 ? -1.0 : 1.0;
        target = kTurnSpeed;
      }
    }
    const double dv = std::clamp(target - v, -kBrake * kDt, kAccel * kDt);
    a_long = dv / kDt;
    a_lat = v * omega;
    v += dv;
    yaw += omega * kDt;
    x += v * std::cos(yaw) * kDt;
    y += v * std::sin(yaw) * kDt;
    if (!turning) left -= v * kDt;
  }
  SplineTrajectory spline(0.0, kKnotDt);
  for (const auto& [R, p] : poses) spline.addControlPose(R, p);
  return spline;
}

}  // namespace

std::string toString(Scenario scenario) {
  switch (scenario) {
    case Scenario::kHandheld: return "handheld";
    case Scenario::kCart: return "cart";
    case Scenario::kFastRotation: return "fast-rotation";
    case Scenario::kCar: return "car";
    case Scenario::kFigure8: return "figure8";
  }
  return "unknown";
}

Scenario parseScenario(const std::string& name) {
  if (name == "handheld") return Scenario::kHandheld;
  if (name == "cart") return Scenario::kCart;
  if (name == "fast_rotation") return Scenario::kFastRotation;
  if (name == "car") return Scenario::kCar;
  if (name == "figure8") return Scenario::kFigure8;
  throw std::invalid_argument("unknown scenario " + name);
}

CameraModelPtr simulatedCamera() {
  Eigen::VectorXd k(9);
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  return std::make_shared<PinholeRadtan>(752, 480, k);
}

SimConfig defaultSimConfig(Scenario scenario) {
  SimConfig config;
  if (scenario == Scenario::kFigure8) {
    config.gyro_turn_on_sigma = 0.01;  // rad/s, ~0.6 deg/s
    config.acc_turn_on_sigma = 0.05;   // m/s^2
  }
  if (scenario == Scenario::kCart) {
    config.ground_density = 1.0;
    config.wall_density = 1.0;
    config.ceiling_density = 0.0;
    config.wall_margin = 8.0;
    config.wall_height = 6.0;
  }
  if (scenario == Scenario::kCar) {
    // KITTI: camera 10 Hz; IMU at 100 Hz (OXTS); a street corridor instead
    // of a room; forward camera pitched slightly down.
    config.cam_rate = 10.0;
    config.imu_rate = 100.0;
    config.ground_density = config.wall_density = config.ceiling_density = 0.0;
    config.corridor_density = 4.0;
    config.corridor_road_density = 3.0;
    config.max_depth = 60.0;
    config.R_IC = forwardCameraRotation(5.0 * M_PI / 180.0);
  }
  return config;
}

TrajectoryPtr makeScenario(Scenario scenario, double duration, uint64_t seed) {
  switch (scenario) {
    case Scenario::kHandheld: return std::make_shared<SplineTrajectory>(makeHandheld(duration, seed));
    case Scenario::kCart: return std::make_shared<SplineTrajectory>(makeCart(duration, seed));
    case Scenario::kFastRotation: return std::make_shared<SplineTrajectory>(makeFastRotation(duration, seed));
    case Scenario::kCar: return std::make_shared<SplineTrajectory>(makeCar(duration, seed));
    case Scenario::kFigure8: {
      auto figure8 = std::make_shared<LissajousTrajectory>(LissajousConfig{}, duration);
      return std::make_shared<TimeWarpedTrajectory>(figure8, kFigure8Static, kFigure8Ramp);
    }
  }
  throw std::invalid_argument("unknown scenario");
}

}  // namespace msckf
