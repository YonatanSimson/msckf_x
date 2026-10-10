// Runs static initialization over a EuRoC-format sequence (EuRoC, TUM-VI) and
// compares roll and pitch with ground truth wherever the IMU is stationary.
//
//   static_init_eval --euroc DIR --gt FILE [options]
//
// Options:
//   --imu-yaml FILE    Kalibr imu.yaml with the noise densities (default: ImuNoise defaults)
//   --window S         stationary window length (default 1.0)
//   --step S           spacing of window end times (default 0.5)
//   --max-acc-std X    stationarity threshold, m/s^2 (default 0.1)
//   --acc-bias-sigma X accelerometer bias prior sigma per axis, m/s^2 (default 0.05; prior mean 0)
//   --out FILE         per-window CSV (default static_init.csv)
//
// The ground truth (EuRoC CSV: t_ns, p, q_wxyz; TUM-VI's dso/gt_imu.csv has the
// same layout) must give IMU poses in a gravity-aligned frame (z up). Yaw is
// the initializer's gauge, so only the tilt is compared: the angle between the
// estimated and true "up" in the IMU frame, and its 2-dof NEES.

#include "msckf/calib/kalibr_io.h"
#include "msckf/eval/trajectory.h"
#include "msckf/init/static_init.h"
#include "msckf/io/euroc.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace msckf;

namespace {

void usage() {
  std::cerr << "usage: static_init_eval --euroc DIR --gt FILE [--imu-yaml FILE] [--window S] "
               "[--step S] [--max-acc-std X] [--acc-bias-sigma X] [--out FILE]\n";
}

// EuRoC's state_groundtruth_estimate0 also carries v, b_w, b_a (17 columns).
struct GtBias {
  double t;
  Eigen::Vector3d bg, ba;
};

std::vector<GtBias> readGtBiases(const std::string& path) {
  std::ifstream f(path);
  std::vector<GtBias> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (char& c : line) {
      if (c == ',') c = ' ';
    }
    std::istringstream ss(line);
    std::vector<double> v;
    double x;
    while (ss >> x) v.push_back(x);
    if (v.size() < 17) return {};
    out.push_back({v[0] * 1e-9, Eigen::Vector3d(v[11], v[12], v[13]),
                   Eigen::Vector3d(v[14], v[15], v[16])});
  }
  return out;
}

const GtBias* nearest(const std::vector<GtBias>& b, double t) {
  if (b.empty()) return nullptr;
  auto it = std::lower_bound(b.begin(), b.end(), t,
                             [](const GtBias& g, double tt) { return g.t < tt; });
  if (it == b.end()) --it;
  if (it != b.begin() && std::abs((it - 1)->t - t) < std::abs(it->t - t)) --it;
  return std::abs(it->t - t) < 0.05 ? &*it : nullptr;
}

double tiltErrorDeg(const ImuInitState& init, const Sophus::SE3d& T_gt) {
  const Eigen::Vector3d z = init.nav.R_GI.inverse() * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d z_gt = T_gt.so3().inverse() * Eigen::Vector3d::UnitZ();
  return std::atan2(z.cross(z_gt).norm(), z.dot(z_gt)) * 180.0 / M_PI;
}

}  // namespace

int main(int argc, char** argv) {
  std::string euroc, gt_path, imu_yaml, out = "static_init.csv";
  StaticInitConfig cfg;
  double step = 0.5;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--euroc") euroc = next();
      else if (arg == "--gt") gt_path = next();
      else if (arg == "--imu-yaml") imu_yaml = next();
      else if (arg == "--window") cfg.window = std::stod(next());
      else if (arg == "--step") step = std::stod(next());
      else if (arg == "--max-acc-std") cfg.max_acc_std = std::stod(next());
      else if (arg == "--acc-bias-sigma") cfg.acc_bias_sigma = std::stod(next());
      else if (arg == "--out") out = next();
      else if (arg == "-h" || arg == "--help") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("unknown argument " + arg);
      }
    }
    if (euroc.empty() || gt_path.empty()) {
      usage();
      return 2;
    }
    if (!imu_yaml.empty()) cfg.noise = calib::loadKalibrImu(imu_yaml);

    const auto imu = io::readEurocImu(euroc + "/mav0/imu0/data.csv");
    const auto gt = eval::readEurocGroundTruth(gt_path);
    if (imu.empty() || gt.empty()) throw std::runtime_error("empty IMU or ground truth");
    const auto gt_bias = readGtBiases(gt_path);

    std::ofstream csv(out);
    csv << "t_end,acc_std,stationary,has_gt,tilt_err_deg,tilt_nees,gravity_norm_err,"
           "bg_err,bg_nees,ba_gt_norm,tilt_err_with_gt_ba_deg,bg_x,bg_y,bg_z,bg_gt_x,bg_gt_y,bg_gt_z\n"
        << std::setprecision(10);
    int with_bias = 0;
    double sum_bg_err = 0.0, sum_bg_nees = 0.0, sum_ba = 0.0, sum_err_oracle = 0.0;
    int windows = 0, stationary = 0, compared = 0;
    double sum_err = 0.0, max_err = 0.0, sum_nees = 0.0;
    std::vector<double> errs;
    for (double t_end = imu.front().t + cfg.window; t_end <= imu.back().t; t_end += step) {
      ++windows;
      StationarityCheck check;
      const auto init = staticInitialize(imu, t_end, cfg, std::nullopt, &check);
      csv << std::fixed << std::setprecision(6) << t_end << std::defaultfloat << std::setprecision(10) << ',' << check.acc_std << ',' << check.stationary;
      Sophus::SE3d T_gt;
      if (!init || !eval::interpolatePose(gt, t_end, 0.05, T_gt)) {
        csv << ",0,,," << check.gravity_error << ",,,,,,,,,,\n";
        stationary += init.has_value();
        continue;
      }
      ++stationary;
      ++compared;
      const Eigen::Vector3d z = init->nav.R_GI.inverse() * Eigen::Vector3d::UnitZ();
      const Eigen::Vector3d z_gt = T_gt.so3().inverse() * Eigen::Vector3d::UnitZ();
      const Eigen::Vector3d dtheta = z_gt.cross(z);  // tilt error, first order
      Eigen::Matrix<double, 3, 2> B;
      B.col(0) = z.unitOrthogonal();
      B.col(1) = z.cross(B.col(0));
      const Eigen::Vector2d e = B.transpose() * dtheta;
      const Eigen::Matrix2d C = B.transpose() * init->P.block<3, 3>(imu_err::kTheta, imu_err::kTheta) * B;
      const double nees = e.dot(C.ldlt().solve(e));
      const double err_deg = std::atan2(z.cross(z_gt).norm(), z.dot(z_gt)) * 180.0 / M_PI;
      sum_err += err_deg;
      max_err = std::max(max_err, err_deg);
      sum_nees += nees;
      errs.push_back(err_deg);
      csv << ",1," << err_deg << ',' << nees << ',' << check.gravity_error;
      if (const GtBias* b = nearest(gt_bias, t_end)) {
        // Gyro bias against ground truth; tilt again with the true b_a as the
        // prior, to separate accelerometer-bias error from everything else.
        const Eigen::Vector3d dbg = b->bg - init->gyro_bias;
        const Eigen::Matrix3d Pbg = init->P.block<3, 3>(imu_err::kBg, imu_err::kBg);
        const double bg_nees = dbg.dot(Pbg.ldlt().solve(dbg));
        StaticInitConfig oracle = cfg;
        oracle.acc_bias_prior = b->ba;
        const auto init_oracle = staticInitialize(imu, t_end, oracle);
        const double err_oracle = init_oracle ? tiltErrorDeg(*init_oracle, T_gt) : NAN;
        ++with_bias;
        sum_bg_err += dbg.norm();
        sum_bg_nees += bg_nees;
        sum_ba += b->ba.norm();
        sum_err_oracle += err_oracle;
        csv << ',' << dbg.norm() << ',' << bg_nees << ',' << b->ba.norm() << ',' << err_oracle;
        for (int k = 0; k < 3; ++k) csv << ',' << init->gyro_bias[k];
        for (int k = 0; k < 3; ++k) csv << ',' << b->bg[k];
        csv << '\n';
      } else {
        csv << ",,,,,,,,,\n";
      }
    }
    std::sort(errs.begin(), errs.end());
    std::cout << std::fixed << std::setprecision(3) << windows << " windows of " << cfg.window
              << " s, " << stationary << " stationary, " << compared << " with ground truth\n";
    if (compared > 0) {
      std::cout << "  tilt error: mean " << sum_err / compared << " deg, median "
                << errs[errs.size() / 2] << " deg, max " << max_err << " deg\n"
                << "  tilt NEES (2 dof): mean " << sum_nees / compared << '\n';
    }
    if (with_bias > 0) {
      std::cout << std::setprecision(5) << "  gyro bias error: mean " << sum_bg_err / with_bias
                << " rad/s, NEES (3 dof) mean " << std::setprecision(3)
                << sum_bg_nees / with_bias << '\n'
                << "  ground-truth |b_a|: mean " << sum_ba / with_bias
                << " m/s^2; tilt error with the true b_a as prior: mean "
                << sum_err_oracle / with_bias << " deg\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "static_init_eval: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
