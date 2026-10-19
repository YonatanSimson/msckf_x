// Writes one simulated run to files (user's simulator plan, 5: output files),
// for python/play_sim.py and any tool outside the C++ build.
//
//   sim_export [--scenario handheld|cart|fast_rotation|car|figure8] [--duration S] [--seed N] --out DIR
//
// Output in --out:
//   imu.csv          t, gyro (rad/s), accelerometer (m/s^2): measured, with noise and bias
//   groundtruth.csv  per IMU sample: t, p_GI, q_GI (Hamilton w, x, y, z), v_GI, gyro bias, accelerometer bias
//   features.csv     per observation: camera time, landmark id, noisy and exact pixel
//   landmarks.csv    id, position in G
//   config.yaml      scenario, seed, rates, noise densities, turn-on sigmas, camera, extrinsic, time offset

#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

struct Args {
  std::string scenario = "figure8", out;
  double duration = 60.0;
  uint64_t seed = 1;
};

std::ofstream openCsv(const fs::path& path, const std::string& header) {
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("cannot write " + path.string());
  csv << header << '\n' << std::setprecision(12);
  return csv;
}

std::ostream& operator<<(std::ostream& os, const Eigen::Vector3d& v) {
  return os << v.x() << ',' << v.y() << ',' << v.z();
}

void writeImu(const fs::path& dir, const SimData& d) {
  std::ofstream csv = openCsv(dir / "imu.csv", "t,wx,wy,wz,ax,ay,az");
  for (const ImuSample& s : d.imu) csv << s.t << ',' << s.gyro << ',' << s.acc << '\n';
}

void writeGroundTruth(const fs::path& dir, const SimData& d) {
  std::ofstream csv = openCsv(dir / "groundtruth.csv", "t,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz");
  for (const GroundTruth& g : d.imu_truth) {
    const Eigen::Quaterniond q = g.state.R_GI.unit_quaternion();
    csv << g.t << ',' << g.state.p_GI << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
        << g.state.v_GI << ',' << g.gyro_bias << ',' << g.acc_bias << '\n';
  }
}

void writeFeatures(const fs::path& dir, const SimData& d) {
  std::ofstream csv = openCsv(dir / "features.csv", "t,id,u,v,u_true,v_true");
  for (const CameraFrame& f : d.frames) {
    for (const Observation& o : f.observations) {
      csv << f.t_cam << ',' << o.landmark_id << ',' << o.uv.x() << ',' << o.uv.y() << ',' << o.uv_true.x() << ','
          << o.uv_true.y() << '\n';
    }
  }
}

void writeLandmarks(const fs::path& dir, const SimData& d) {
  std::ofstream csv = openCsv(dir / "landmarks.csv", "id,x,y,z");
  for (const Landmark& l : d.landmarks) csv << l.id << ',' << l.p_G << '\n';
}

void writeConfig(const fs::path& dir, const Args& a, const Simulator& sim) {
  const SimConfig& c = sim.config();
  const ImuNoise& n = c.imu_noise;
  const Eigen::Quaterniond q = c.R_IC.unit_quaternion();
  std::ofstream yaml(dir / "config.yaml");
  yaml << std::setprecision(12) << "scenario: " << a.scenario << "\nseed: " << c.seed << "\nduration: " << c.duration
       << "\nimu_rate: " << c.imu_rate << "\ncam_rate: " << c.cam_rate << "\ngyro_noise: " << n.gyro_noise
       << "\nacc_noise: " << n.acc_noise << "\ngyro_bias_walk: " << n.gyro_bias_walk
       << "\nacc_bias_walk: " << n.acc_bias_walk << "\ngyro_turn_on_sigma: " << c.gyro_turn_on_sigma
       << "\nacc_turn_on_sigma: " << c.acc_turn_on_sigma << "\npixel_noise: " << c.pixel_noise
       << "\ntime_offset: " << c.time_offset << "\ncamera:\n  model: " << sim.camera().name()
       << "\n  width: " << sim.camera().width() << "\n  height: " << sim.camera().height() << "\n  params: ["
       << sim.camera().params().transpose().format(Eigen::IOFormat(12, Eigen::DontAlignCols, ", ")) << "]"
       << "\nq_IC: [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]  # Hamilton w, x, y, z"
       << "\np_IC: [" << c.p_IC.x() << ", " << c.p_IC.y() << ", " << c.p_IC.z() << "]\n";
}

int run(const Args& a) {
  if (a.out.empty()) throw std::invalid_argument("--out is required");
  const Scenario scenario = parseScenario(a.scenario);
  SimConfig c = defaultSimConfig(scenario);
  c.duration = a.duration;
  c.seed = a.seed;
  const Simulator sim(c, makeScenario(scenario, c.duration + 1.0, a.seed), simulatedCamera());
  const SimData d = sim.run();
  const fs::path dir(a.out);
  fs::create_directories(dir);
  writeImu(dir, d);
  writeGroundTruth(dir, d);
  writeFeatures(dir, d);
  writeLandmarks(dir, d);
  writeConfig(dir, a, sim);
  std::cout << a.scenario << ": " << d.imu.size() << " IMU samples, " << d.frames.size() << " images, "
            << d.landmarks.size() << " landmarks -> " << dir.string() << "\n";
  return 0;
}

Args parseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--scenario") a.scenario = next();
    else if (arg == "--duration") a.duration = std::stod(next());
    else if (arg == "--seed") a.seed = std::stoull(next());
    else if (arg == "--out") a.out = next();
    else throw std::invalid_argument("unknown option " + arg);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseArgs(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "sim_export: " << e.what() << "\n";
    return 2;
  }
}
