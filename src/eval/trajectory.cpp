#include "msckf/eval/trajectory.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace msckf::eval {

namespace {

std::ifstream openInput(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  return in;
}

bool isBlankOrComment(const std::string& line) {
  const auto pos = line.find_first_not_of(" \t\r");
  return pos == std::string::npos || line[pos] == '#';
}

void sortByTime(Trajectory& traj) {
  std::sort(traj.begin(), traj.end(),
            [](const StampedPose& a, const StampedPose& b) { return a.t < b.t; });
}

}  // namespace

Trajectory readTum(const std::string& path) {
  std::ifstream in = openInput(path);
  Trajectory out;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (isBlankOrComment(line)) continue;
    std::istringstream ss(line);
    double t, x, y, z, qx, qy, qz, qw;
    if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected 8 numbers");
    }
    const Eigen::Quaterniond q(qw, qx, qy, qz);
    out.push_back({t, Sophus::SE3d(q.normalized(), Eigen::Vector3d(x, y, z))});
  }
  sortByTime(out);
  return out;
}

void writeTum(const std::string& path, const Trajectory& trajectory) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "# t tx ty tz qx qy qz qw\n" << std::setprecision(17);
  for (const StampedPose& s : trajectory) {
    const Eigen::Vector3d& p = s.T_GI.translation();
    const Eigen::Quaterniond& q = s.T_GI.unit_quaternion();
    out << s.t << ' ' << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << q.x() << ' ' << q.y()
        << ' ' << q.z() << ' ' << q.w() << '\n';
  }
}

Trajectory readEurocGroundTruth(const std::string& path) {
  std::ifstream in = openInput(path);
  Trajectory out;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (isBlankOrComment(line)) continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    long long t_ns;
    double x, y, z, qw, qx, qy, qz;
    if (!(ss >> t_ns >> x >> y >> z >> qw >> qx >> qy >> qz)) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected EuRoC GT row");
    }
    const Eigen::Quaterniond q(qw, qx, qy, qz);
    out.push_back({t_ns * 1e-9, Sophus::SE3d(q.normalized(), Eigen::Vector3d(x, y, z))});
  }
  sortByTime(out);
  return out;
}

Trajectory readTrajectory(const std::string& path, const std::string& format) {
  if (format == "tum") return readTum(path);
  if (format == "euroc") return readEurocGroundTruth(path);
  throw std::invalid_argument("unknown trajectory format '" + format + "'");
}

bool interpolatePose(const Trajectory& trajectory, double t, double max_gap, Sophus::SE3d& T) {
  // Timestamps that agree to within a microsecond are the same sample: files
  // store seconds and nanoseconds, so equal stamps rarely compare equal.
  constexpr double kSameStamp = 1e-6;
  if (trajectory.empty() || t < trajectory.front().t - kSameStamp ||
      t > trajectory.back().t + kSameStamp) {
    return false;
  }
  const auto it = std::lower_bound(trajectory.begin(), trajectory.end(), t,
                                   [](const StampedPose& s, double v) { return s.t < v; });
  if (it != trajectory.end() && it->t - t <= kSameStamp) {
    T = it->T_GI;
    return true;
  }
  if (it != trajectory.begin() && t - (it - 1)->t <= kSameStamp) {
    T = (it - 1)->T_GI;
    return true;
  }
  if (it == trajectory.end() || it == trajectory.begin()) return false;
  const StampedPose& b = *it;
  const StampedPose& a = *(it - 1);
  if (b.t - a.t > max_gap + kSameStamp) return false;
  const double alpha = (t - a.t) / (b.t - a.t);
  const Sophus::SO3d R =
      a.T_GI.so3() * Sophus::SO3d::exp(alpha * (a.T_GI.so3().inverse() * b.T_GI.so3()).log());
  const Eigen::Vector3d p = (1.0 - alpha) * a.T_GI.translation() + alpha * b.T_GI.translation();
  T = Sophus::SE3d(R, p);
  return true;
}

Association associate(const Trajectory& estimate, const Trajectory& ground_truth, double max_gap) {
  Association out;
  for (size_t i = 0; i < estimate.size(); ++i) {
    Sophus::SE3d T_gt;
    if (!interpolatePose(ground_truth, estimate[i].t, max_gap, T_gt)) continue;
    out.t.push_back(estimate[i].t);
    out.est.push_back(estimate[i].T_GI);
    out.gt.push_back(T_gt);
    out.est_index.push_back(i);
  }
  return out;
}

}  // namespace msckf::eval
