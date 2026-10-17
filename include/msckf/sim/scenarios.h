#pragma once

#include "msckf/camera/camera_model.h"
#include "msckf/sim/spline.h"

#include <cstdint>
#include <string>

namespace msckf {

enum class Scenario {
  kHandheld,      // rich excitation in a room
  kCart,          // slow, planar, pushed (+-0.15 m/s surges at 0.6 Hz), exact stops, U-turns
  kFastRotation,  // rotation rates up to ~5 rad/s, small translation
  kCar,           // KITTI-like driving: 8-12 m/s cruise, braking for 90 deg turns, rolling start
};

std::string toString(Scenario scenario);

/// Scenario from its command-line name: handheld, cart, fast_rotation or car.
/// @throws std::invalid_argument for any other name.
Scenario parseScenario(const std::string& name);

/// The camera every simulation uses: EuRoC cam0 (pinhole, radial-tangential, 752 x 480).
CameraModelPtr simulatedCamera();

struct SimConfig;

// Simulator settings that suit the scenario: handheld and fast rotation are
// indoor rooms with a ceiling; the cart is outdoors (ground and distant walls,
// no ceiling).
SimConfig defaultSimConfig(Scenario scenario);

// IMU-frame convention for all scenarios: x forward, y left, z up.
// The spline covers at least [0, duration].
SplineTrajectory makeScenario(Scenario scenario, double duration, uint64_t seed);

}  // namespace msckf
