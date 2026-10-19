// Stage 0 gate: our Fisheye624 agrees with Project Aria's own implementation
// (projectaria_tools, Apache 2.0) on a LaMAria calibration to < 1e-9 px and
// < 1e-9 in bearing. The reference samples come from
// python/aria_fisheye_fixture.py (tests/data/fisheye624_aria.csv); nothing
// from projectaria_tools is linked here.

#include <gtest/gtest.h>

#include "msckf/camera/fisheye624.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace msckf {
namespace {

struct Sample {
  Eigen::Vector3d point;
  Eigen::Vector2d pixel;
  Eigen::Vector3d bearing;
};

struct Fixture {
  Eigen::VectorXd params;
  int width = 0, height = 0;
  std::vector<Sample> samples;
};

Fixture readFixture(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  Fixture f;
  std::string line, word;
  std::getline(in, line);  // "# params ..."
  std::istringstream ps(line.substr(line.find("params") + 6));
  std::vector<double> p;
  for (double x; ps >> x;) p.push_back(x);
  f.params = Eigen::Map<Eigen::VectorXd>(p.data(), static_cast<Eigen::Index>(p.size()));
  std::getline(in, line);  // "# width height"
  std::istringstream ws(line.substr(1));
  ws >> f.width >> f.height;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::vector<double> v;
    while (std::getline(ss, word, ',')) v.push_back(std::stod(word));
    if (v.size() != 8) throw std::runtime_error("bad fixture row: " + line);
    f.samples.push_back({{v[0], v[1], v[2]}, {v[3], v[4]}, {v[5], v[6], v[7]}});
  }
  return f;
}

TEST(Fisheye624AriaCrossCheck, ProjectionAndUnprojectionAgree) {
  const Fixture f = readFixture(std::string(MSCKF_TEST_DATA_DIR) + "/fisheye624_aria.csv");
  ASSERT_GT(f.samples.size(), 1000u);
  const Fisheye624 cam(f.width, f.height, f.params);
  double max_px = 0.0, max_bearing = 0.0;
  for (const Sample& s : f.samples) {
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam.project(s.point, uv)) << s.point.transpose();
    max_px = std::max(max_px, (uv - s.pixel).norm());
    Eigen::Vector3d b;
    ASSERT_TRUE(cam.unproject(s.pixel, b)) << s.pixel.transpose();
    max_bearing = std::max(max_bearing, (b - s.bearing).norm());
  }
  std::cout << "[ info ] " << f.samples.size() << " samples, max projection difference " << max_px
            << " px, max bearing difference " << max_bearing << "\n";
  EXPECT_LT(max_px, 1e-9);
  EXPECT_LT(max_bearing, 1e-9);
}

}  // namespace
}  // namespace msckf
