#pragma once

#include "msckf/sim/spline.h"

#include <cstdint>
#include <string>

namespace msckf {

enum class Scenario {
  kHandheld,      // rich excitation in a room
  kCart,          // slow, planar, long constant-speed legs, exact stops, U-turns
  kFastRotation,  // rotation rates up to ~5 rad/s, small translation
};

std::string toString(Scenario scenario);

struct SimConfig;

// Simulator settings that suit the scenario: handheld and fast rotation are
// indoor rooms with a ceiling; the cart is outdoors (ground and distant walls,
// no ceiling).
SimConfig defaultSimConfig(Scenario scenario);

// IMU-frame convention for all scenarios: x forward, y left, z up.
// The spline covers at least [0, duration].
SplineTrajectory makeScenario(Scenario scenario, double duration, uint64_t seed);

}  // namespace msckf
