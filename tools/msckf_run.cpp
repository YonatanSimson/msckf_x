// MSCKF on a recorded sequence (plan, Stage 3: "MSCKF on real data").
//
//   msckf_run --euroc DIR --tracks tracks.csv --out DIR [--cam cam0] [--camchain YAML] [--imu-yaml YAML]
//             [--gt CSV --bias zero|gt] [--imu-noise-scale K] [--pixel-sigma PX] [--clones N] [--no-fej]
//
// Sequences in EuRoC layout (EuRoC, TUM-VI, LaMAria and KITTI conversions)
// with the front end's tracks (tracker_eval tracks). Initializes with the
// Stage 1b pipeline (initializeDynamic, retried every 0.5 s until it
// succeeds), then runs the filter on every later image.
//
// Biases: zero, or with --bias gt EuRoC's ground-truth biases at the start
// (the initializer's "prior calibration"; the filter estimates them after).
// The extrinsic and time offset start at the calibration and are estimated.
//
// Output in --out:
//   est.txt     IMU pose per image (TUM format, IMU clock), for msckf_eval
//   frames.csv  per image: features, used, rejections, rows, NIS, time
//   run.csv     initialization time and kind, images processed, divergence

#include "msckf/calib/kalibr_io.h"
#include "msckf/eval/trajectory.h"
#include "msckf/filter/msckf.h"
#include "msckf/init/dynamic_init.h"
#include "msckf/io/euroc.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

struct Args {
  std::string euroc, cam = "cam0", camchain, tracks, imu_yaml, gt, bias = "zero", out = "msckf_run_out";
  double imu_noise_scale = 1.0, pixel_sigma = 1.0;
  int clones = filter::MsckfOptions{}.max_clones;
  bool fej = true;
};

struct Sequence {
  calib::KalibrCamera camera;
  std::vector<ImuSample> imu;
  std::vector<io::TrackedFrame> frames;
  ImuNoise noise;
};

struct Initialization {
  ImuInitState state;
  std::string kind;  // static, in_place, linear, dynamic
};

constexpr double kRetryStep = 0.5;   // s between initialization attempts
constexpr double kMaxPosition = 1e4;  // m; beyond this the filter has diverged

Sequence loadSequence(const Args& a) {
  Sequence s;
  s.camera = calib::loadSequenceCamera(a.euroc, a.cam, a.camchain);
  if (!s.camera.has_T_IC) throw std::runtime_error("camera has no extrinsic (T_cam_imu / T_BS)");
  s.imu = io::readEurocImu((fs::path(a.euroc) / "mav0" / "imu0" / "data.csv").string());
  s.frames = io::readTracks(a.tracks);
  if (s.frames.empty() || s.imu.empty()) throw std::runtime_error("no images or no IMU samples");
  s.noise = a.imu_yaml.empty() ? ImuNoise{} : calib::loadKalibrImu(a.imu_yaml);
  s.noise.gyro_noise *= a.imu_noise_scale;
  s.noise.acc_noise *= a.imu_noise_scale;
  s.noise.gyro_bias_walk *= a.imu_noise_scale;
  s.noise.acc_bias_walk *= a.imu_noise_scale;
  return s;
}

std::vector<InitFrame> initFrames(const Sequence& s) {
  std::vector<InitFrame> out;
  for (const io::TrackedFrame& tf : s.frames) {
    InitFrame& fr = out.emplace_back();
    fr.t = tf.t;
    for (const auto& [id, uv] : tf.uv) {
      Eigen::Vector3d b;
      if (s.camera.model->unproject(uv, b)) fr.bearings.emplace_back(id, b);
    }
  }
  return out;
}

// Ground-truth biases at t (--bias gt), else zero.
std::pair<Eigen::Vector3d, Eigen::Vector3d> priorBiases(const Args& a, double t) {
  if (a.bias == "zero") return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  if (a.bias != "gt") throw std::invalid_argument("--bias " + a.bias);
  const std::vector<io::EurocState> states = io::readEurocStates(a.gt);
  if (states.empty()) throw std::runtime_error("--bias gt needs EuRoC's state ground truth (--gt)");
  const auto it = std::lower_bound(states.begin(), states.end(), t,
                                   [](const io::EurocState& x, double v) { return x.t < v; });
  const io::EurocState& st = it == states.end() ? states.back() : *it;
  return {st.bg, st.ba};
}

std::string initKind(const DynamicInitResult& r) {
  return r.in_place ? "in_place" : r.is_static ? "static" : r.linear_only ? "linear" : "dynamic";
}

// The first successful initialization from the start, one attempt every kRetryStep.
Initialization initialize(const Sequence& s, const Args& a) {
  const std::vector<InitFrame> frames = initFrames(s);
  const double pixels_per_radian = pixelsPerRadian(*s.camera.model);
  DynamicInitConfig cfg;
  cfg.pixels_per_radian = pixels_per_radian;
  cfg.refinement.bearing_sigma = a.pixel_sigma / pixels_per_radian;
  const double t_end = std::min(frames.back().t, s.imu.back().t) - cfg.max_window;
  for (double t0 = std::max(frames.front().t, s.imu.front().t) + 0.5; t0 < t_end; t0 += kRetryStep) {
    const auto [bg, ba] = priorBiases(a, t0);
    const DynamicInitResult r = initializeDynamic(s.imu, frames, t0, s.camera.T_IC, bg, ba, s.noise, cfg);
    if (r.ok) return {r.refinement.state, initKind(r)};
  }
  throw std::runtime_error("initialization failed on the whole sequence");
}

filter::MsckfOptions filterOptions(const Sequence& s, const Args& a) {
  filter::MsckfOptions options;
  options.noise = s.noise;
  options.max_clones = a.clones;
  options.state.fej = a.fej;
  options.update.pixel_noise = a.pixel_sigma;
  return options;
}

// Extrinsic rotation, translation and time offset sigmas: calibrated values.
Eigen::Matrix<double, 7, 7> calibrationCovariance() {
  Eigen::Matrix<double, 7, 1> sd;
  sd << Eigen::Vector3d::Constant(0.5 * M_PI / 180.0), Eigen::Vector3d::Constant(0.01), 0.005;
  return sd.cwiseAbs2().asDiagonal();
}

void writeFrame(std::ostream& csv, double t, const filter::FrameResult& r) {
  const filter::UpdateStats& u = r.update;
  csv << t << ',' << u.features << ',' << u.used << ',' << u.rejected_triangulation << ',' << u.rejected_chi2 << ','
      << u.rows << ',' << u.nis << ',' << r.time_ms << '\n';
}

bool diverged(const filter::FilterState& s) {
  return !s.nav.p_GI.allFinite() || !s.P.allFinite() || s.nav.p_GI.norm() > kMaxPosition;
}

// Runs the filter on every image after the initialization; returns the images
// processed and whether it diverged.
std::pair<int, bool> runFilter(const Sequence& s, const Initialization& init, const Args& a) {
  const Sophus::SE3d& T_IC = s.camera.T_IC;
  filter::Msckf msckf(s.camera.model, init.state, T_IC.so3(), T_IC.translation(), 0.0, calibrationCovariance(),
                      filterOptions(s, a));
  std::ofstream csv(fs::path(a.out) / "frames.csv");
  csv << "t,features,used,rejected_triangulation,rejected_chi2,rows,nis,time_ms\n" << std::setprecision(15);
  eval::Trajectory est;
  size_t imu_i = 0;
  int images = 0;
  bool div = false;
  for (const io::TrackedFrame& tf : s.frames) {
    const double t_img = tf.t + msckf.state().t_d;
    if (t_img <= init.state.t) continue;
    while (imu_i < s.imu.size() && s.imu[imu_i].t <= t_img + 0.01) msckf.addImu(s.imu[imu_i++]);
    if (s.imu.back().t < t_img) break;
    std::vector<filter::FeatureObservation> obs;
    for (const auto& [id, uv] : tf.uv) obs.push_back({id, uv});
    writeFrame(csv, tf.t, msckf.addImage(tf.t, obs));
    ++images;
    const filter::Clone& c = msckf.state().clones().back();
    if ((div = diverged(msckf.state()))) break;
    est.push_back({tf.t + msckf.state().t_d, Sophus::SE3d(c.R_GI, c.p_GI)});
  }
  eval::writeTum((fs::path(a.out) / "est.txt").string(), est);
  return {images, div};
}

int run(const Args& a) {
  if (a.euroc.empty() || a.tracks.empty()) throw std::invalid_argument("--euroc and --tracks are required");
  fs::create_directories(a.out);
  const Sequence s = loadSequence(a);
  const Initialization init = initialize(s, a);
  const auto [images, div] = runFilter(s, init, a);
  std::ofstream(fs::path(a.out) / "run.csv") << "t_init,init,images,diverged\n"
                                              << std::setprecision(15) << init.state.t << ',' << init.kind << ','
                                              << images << ',' << div << '\n';
  std::cout << "initialized at " << init.state.t << " s (" << init.kind << "); " << images << " images"
            << (div ? ", DIVERGED" : "") << "\n";
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
    if (arg == "--euroc") a.euroc = next();
    else if (arg == "--cam") a.cam = next();
    else if (arg == "--camchain") a.camchain = next();
    else if (arg == "--tracks") a.tracks = next();
    else if (arg == "--imu-yaml") a.imu_yaml = next();
    else if (arg == "--gt") a.gt = next();
    else if (arg == "--bias") a.bias = next();
    else if (arg == "--imu-noise-scale") a.imu_noise_scale = std::stod(next());
    else if (arg == "--pixel-sigma") a.pixel_sigma = std::stod(next());
    else if (arg == "--clones") a.clones = std::stoi(next());
    else if (arg == "--no-fej") a.fej = false;
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
    std::cerr << "msckf_run: " << e.what() << "\n";
    return 2;
  }
}
