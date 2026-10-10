#pragma once

#include "msckf/imu/imu_types.h"

#include <string>
#include <vector>

namespace msckf::io {

// EuRoC / TUM-VI IMU CSV (mav0/imu0/data.csv):
// "t_ns, wx, wy, wz, ax, ay, az" with a '#' header. Times in seconds.
std::vector<ImuSample> readEurocImu(const std::string& path);

}  // namespace msckf::io
