#include "msckf/io/euroc.h"

#include <fstream>
#include <map>
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

namespace {

std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

}  // namespace

std::vector<EurocState> readEurocStates(const std::string& path) {
  std::ifstream f(path);
  std::vector<EurocState> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<double> v;
    for (const std::string& tok : splitCsv(line)) v.push_back(std::stod(tok));
    if (v.size() < 17) return {};
    out.push_back({v[0] * 1e-9, Eigen::Vector3d(v[8], v[9], v[10]), Eigen::Vector3d(v[11], v[12], v[13]),
                   Eigen::Vector3d(v[14], v[15], v[16])});
  }
  return out;
}

std::vector<TrackedFrame> readTracks(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::map<int, TrackedFrame> frames;
  std::string line;
  std::getline(f, line);
  while (std::getline(f, line)) {
    const std::vector<std::string> v = splitCsv(line);
    if (v.size() < 5) throw std::runtime_error(path + ": malformed line: " + line);
    TrackedFrame& fr = frames[std::stoi(v[0])];
    fr.t = std::stoll(v[1]) * 1e-9;
    fr.uv.emplace_back(std::stoll(v[2]), Eigen::Vector2d(std::stod(v[3]), std::stod(v[4])));
  }
  std::vector<TrackedFrame> out;
  for (auto& [k, fr] : frames) out.push_back(std::move(fr));
  return out;
}

}  // namespace msckf::io
