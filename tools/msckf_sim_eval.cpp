// Runs the MSCKF on simulated scenarios, Monte Carlo over IMU noise, pixel
// noise and the initial error (plan, Stage 3 metrics: consistency, calibration
// convergence). Every run shares the trajectory (scenario seed 1); run r uses
// noise seed r.
//
//   msckf_sim_eval [--scenario handheld|cart|fast_rotation|car] [--runs N] [--duration S]
//                  [--no-fej] [--no-extrinsic] [--no-time-offset] [--clones N] [--pixel-noise PX]
//                  [--min-parallax DEG] [--init-scale K]
//                  [--depth-stats] [--oracle-jacobians] [--oracle-selection] [--filter-noise-scale K]
//                  [--no-perturb] --out DIR
//
// Output: epochs.csv, one row per run and image: true speed, errors of the newest clone
// (rotation in deg, position in m), velocity error (m/s), pose / orientation /
// position / velocity NEES, calibration errors, observations in the image,
// update statistics and time. With --depth-stats, depth.csv: log(estimated /
// true depth) of every used feature.
// The Stage 3 gate (python/msckf_gates/stage3_sim.py) computes the metrics.

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/eval/nees.h"
#include "msckf/filter/sim_run.h"
#include "msckf/sim/scenarios.h"

#include <Eigen/Cholesky>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

struct Args {
  std::string scenario = "handheld", out = "msckf_sim_out";
  int runs = 20, clones = 11;
  double duration = 60.0, pixel_noise = -1.0, min_parallax_deg = filter::UpdateOptions{}.min_parallax_deg;
  double filter_noise_scale = 1.0;  // filter's pixel sigma / simulated
  double init_scale = 1.0;  // scales every initial sigma (SimRunConfig)
  bool depth_stats = false, oracle_jacobians = false, oracle_selection = false;  // diagnosis, with the true landmarks
  bool fej = true, extrinsic = true, time_offset = true, perturb = true;
};

Scenario parseScenario(const std::string& s) {
  if (s == "handheld") return Scenario::kHandheld;
  if (s == "cart") return Scenario::kCart;
  if (s == "fast_rotation") return Scenario::kFastRotation;
  if (s == "car") return Scenario::kCar;
  throw std::invalid_argument("unknown scenario " + s);
}

int run(const Args& a) {
  const Scenario scenario = parseScenario(a.scenario);
  Eigen::VectorXd k(9);  // EuRoC cam0
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  const auto camera = std::make_shared<PinholeRadtan>(752, 480, k);
  filter::MsckfOptions options;
  options.state.fej = a.fej;
  options.state.estimate_extrinsic = a.extrinsic;
  options.state.estimate_time_offset = a.time_offset;
  options.max_clones = a.clones;
  options.update.min_parallax_deg = a.min_parallax_deg;

  fs::create_directories(a.out);
  std::ofstream csv(fs::path(a.out) / "epochs.csv");
  csv << "run,t,speed_mps,rot_err_deg,pos_err_m,pos_err_z_m,vel_err_mps,nees_pose,nees_ori,nees_pos,nees_vel,nees_bg,nees_ba,"
         "ext_rot_err_deg,"
         "ext_pos_err_m,td_err_ms,observations,features,used,rejected_triangulation,rejected_chi2,rows,nis,time_ms\n"
      << std::setprecision(10);
  std::ofstream depth_csv;
  if (a.depth_stats) {
    depth_csv.open(fs::path(a.out) / "depth.csv");
    depth_csv << "run,t,depth_log_ratio\n";
  }
  for (int r = 1; r <= a.runs; ++r) {
    SimConfig c = defaultSimConfig(scenario);
    c.duration = a.duration + 2.0;
    c.seed = static_cast<uint64_t>(r);
    if (a.pixel_noise >= 0.0) c.pixel_noise = a.pixel_noise;
    options.update.pixel_noise = std::max(c.pixel_noise, 0.1) * a.filter_noise_scale;
    const Simulator sim(c, makeScenario(scenario, c.duration + 1.0, 1), camera);
    const SimData data = sim.run();
    options.update.oracle_jacobians = a.oracle_jacobians;
    options.update.oracle_selection = a.oracle_selection;
    if (a.depth_stats || a.oracle_jacobians || a.oracle_selection) {
      options.update.landmark_oracle = [&data](int64_t id, Eigen::Vector3d* p) {
        if (id < 0 || id >= static_cast<int64_t>(data.landmarks.size())) return false;
        *p = data.landmarks[static_cast<size_t>(id)].p_G;  // ids are indices
        return true;
      };
    }
    filter::SimRunConfig rc;
    rc.seed = static_cast<uint64_t>(1000 + r);
    rc.perturb = a.perturb;
    for (double* s : {&rc.sigma_theta, &rc.sigma_p, &rc.sigma_v, &rc.sigma_bg, &rc.sigma_ba, &rc.sigma_ext_theta,
                      &rc.sigma_ext_p, &rc.sigma_td}) {
      *s *= a.init_scale;
    }
    const auto epochs = filter::runFilterOnSimulation(sim, camera, data, options, rc);
    options.update.landmark_oracle = nullptr;  // it refers to this run's data
    double sum_pose = 0.0;
    for (const auto& e : epochs) {
      const eval::PoseNees n = eval::poseNees(e.T_est, e.cov_pose, e.T_true);
      const Eigen::Vector3d dv = e.v_true - e.v_est;
      const double nees_v = dv.dot(e.cov_v.ldlt().solve(dv));
      const Eigen::Vector3d dth = (e.T_est.so3().inverse() * e.T_true.so3()).log();
      sum_pose += n.pose;
      if (a.depth_stats) {
        for (double d : e.update.depth_log_ratio) depth_csv << r << ',' << e.t << ',' << d << '\n';
      }
      csv << r << ',' << e.t << ',' << e.v_true.norm() << ',' << dth.norm() * 180.0 / M_PI << ','
          << (e.T_true.translation() - e.T_est.translation()).norm() << ','
          << e.T_true.translation().z() - e.T_est.translation().z() << ',' << dv.norm() << ',' << n.pose << ','
          << n.orientation << ',' << n.position << ',' << nees_v << ',' << e.nees_bg << ',' << e.nees_ba << ',' << e.ext_theta_err.norm() * 180.0 / M_PI
          << ',' << e.ext_p_err.norm() << ',' << e.td_err * 1e3 << ',' << e.observations << ',' << e.update.features << ','
          << e.update.used << ',' << e.update.rejected_triangulation << ',' << e.update.rejected_chi2 << ','
          << e.update.rows << ',' << e.update.nis << ',' << e.time_ms << '\n';
    }
    if (!epochs.empty()) {
      const auto& l = epochs.back();
      std::cout << "run " << r << ": " << epochs.size() << " images, final pos err "
                << (l.T_true.translation() - l.T_est.translation()).norm() << " m, mean pose NEES "
                << sum_pose / static_cast<double>(epochs.size()) << '\n';
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--scenario") a.scenario = next();
      else if (arg == "--runs") a.runs = std::stoi(next());
      else if (arg == "--duration") a.duration = std::stod(next());
      else if (arg == "--clones") a.clones = std::stoi(next());
      else if (arg == "--pixel-noise") a.pixel_noise = std::stod(next());
      else if (arg == "--min-parallax") a.min_parallax_deg = std::stod(next());
      else if (arg == "--filter-noise-scale") a.filter_noise_scale = std::stod(next());
      else if (arg == "--init-scale") a.init_scale = std::stod(next());
      else if (arg == "--depth-stats") a.depth_stats = true;
      else if (arg == "--oracle-jacobians") a.oracle_jacobians = true;
      else if (arg == "--oracle-selection") a.oracle_selection = true;
      else if (arg == "--no-fej") a.fej = false;
      else if (arg == "--no-extrinsic") a.extrinsic = false;
      else if (arg == "--no-time-offset") a.time_offset = false;
      else if (arg == "--no-perturb") a.perturb = false;
      else if (arg == "--out") a.out = next();
      else throw std::invalid_argument("unknown argument " + arg);
    }
    return run(a);
  } catch (const std::exception& e) {
    std::cerr << "msckf_sim_eval: " << e.what() << '\n';
    return 1;
  }
}
