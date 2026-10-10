#include "msckf/io/euroc.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace msckf::io {

std::vector<ImuSample> readEurocImu(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<ImuSample> imu;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (char& c : line) {
      if (c == ',') c = ' ';
    }
    std::istringstream ss(line);
    long long t_ns;
    ImuSample s;
    if (!(ss >> t_ns >> s.gyro.x() >> s.gyro.y() >> s.gyro.z() >> s.acc.x() >> s.acc.y() >>
          s.acc.z())) {
      throw std::runtime_error(path + ": malformed line: " + line);
    }
    s.t = t_ns * 1e-9;
    imu.push_back(s);
  }
  return imu;
}

}  // namespace msckf::io
