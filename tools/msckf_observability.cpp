// Observability of the filter's linearization (plan, Stage 3 metrics:
// "numerical rank and |O N| on simulation, FEJ on vs off").
//
//   msckf_observability [--scenario handheld|cart|fast_rotation|car] [--duration S] [--seed N]
//                       [--window K] [--stride S] [--features L] --out DIR
//
// Runs the MSCKF on one simulated run, with FEJ and without, and for windows of
// K images (one every --stride seconds) stacks the observability matrix of the
// IMU state and L landmarks seen in at least 3 images of the window:
//   O = [H_1; H_2 Phi(2,1); ...; H_K Phi(K,1)],
// with each Phi the transition the filter applied to P (FilterState::Phi_last)
// and each H from linearizeFeature at the clone's first estimate (FEJ) or at
// the updated estimate of its image (no FEJ: an approximation, the filter
// uses the clone's estimate at the time of the update), with a fixed estimate
// of each landmark. N is
// the analytic nullspace of the system at the first image: global translation
// and yaw about gravity (body-frame dtheta = R^T z, dp = z x p, dv = z x v, a
// landmark moved by z x p_f). With FEJ, O N = 0 and O loses exactly those 4
// directions; without FEJ, yaw gains spurious information.
//
// Output: windows.csv, one row per window and FEJ setting: the 6 smallest
// singular values of O (columns scaled to unit norm) over the largest, the
// nullity (columns minus singular values above 1e-8 of the largest: the
// unobservable directions), and |O n| / (|O|_F |n|) for each nullspace
// direction. Windows that cannot be linearized are logged and skipped.

#include "msckf/filter/feature_update.h"
#include "msckf/filter/observability.h"
#include "msckf/filter/sim_run.h"
#include "msckf/sim/scenarios.h"

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

struct Args {
  std::string scenario = "handheld", out = "observability_out";
  double duration = 30.0, stride = 5.0;
  int seed = 1, window = 20, features = 20;
};

// A filter setting and where it evaluates each image's H.
struct Variant {
  bool fej;
  NavState filter::SimEpoch::*lin;
};
constexpr Variant kVariants[] = {{true, &filter::SimEpoch::nav_prior}, {false, &filter::SimEpoch::nav_post}};

// One filter run on the simulation.
struct FilterRun {
  const SimConfig& config;
  const SimData& data;
  const CameraModel& camera;
  std::vector<filter::SimEpoch> epochs;
  Variant variant;
};

struct WindowLandmark {
  int64_t id = 0;
  std::vector<size_t> seen;  // window images (0 .. K-1) that observe it
  std::vector<Eigen::Vector2d> uv;
};

struct Window {
  std::vector<size_t> epochs;  // into the run's epochs, consecutive
  std::vector<WindowLandmark> landmarks;
};

// What O is built from: clones at the linearization points, the landmark
// estimates, and Phi(k, 1) per image.
struct WindowModel {
  filter::FilterState state;
  std::vector<AnchoredPoint> landmarks;
  std::vector<Matrix15d> Phi;
};

struct WindowResult {
  Eigen::VectorXd smallest;  // 6 smallest singular values / largest, columns scaled
  int unobservable = 0;
  Eigen::Vector4d on = Eigen::Vector4d::Zero();  // |O n| / (|O|_F |n|) per nullspace direction
};

constexpr double kLandmarkSigma = 0.05;  // m, error of each landmark's fixed estimate
constexpr double kRankThreshold = 1e-8;  // singular value / largest

// Images k = start .. start + K - 1 and up to L landmarks seen in at least 3
// of them (the most observed first), as the MSCKF's partial tracks.
Window makeWindow(const FilterRun& run, size_t start, const Args& a) {
  Window w;
  std::map<int64_t, WindowLandmark> by_id;
  const std::vector<CameraFrame>& frames = run.data.frames;
  for (size_t k = start; k < start + static_cast<size_t>(a.window) && k < run.epochs.size(); ++k) {
    size_t f = 0;
    while (f < frames.size() && frames[f].t_cam < run.epochs[k].t - 1e-9) ++f;
    for (const Observation& o : frames[f].observations) {
      WindowLandmark& l = by_id[o.landmark_id];
      l.id = o.landmark_id;
      l.seen.push_back(w.epochs.size());
      l.uv.push_back(o.uv);
    }
    w.epochs.push_back(k);
  }
  for (auto& [id, l] : by_id) {
    if (l.seen.size() >= 3) w.landmarks.push_back(std::move(l));
  }
  std::stable_sort(w.landmarks.begin(), w.landmarks.end(),
                   [](const WindowLandmark& x, const WindowLandmark& y) { return x.seen.size() > y.seen.size(); });
  if (static_cast<int>(w.landmarks.size()) > a.features) w.landmarks.resize(static_cast<size_t>(a.features));
  return w;
}

// A filter state whose clones are the window's linearization points, so that
// linearizeFeature evaluates each image's H there (fej off: the clones already
// hold the chosen points; the FEJ branch is unit tested).
filter::FilterState cloneWindow(const FilterRun& run, const Window& w) {
  filter::StateOptions options;
  options.fej = false;
  options.estimate_extrinsic = false;
  options.estimate_time_offset = false;
  ImuInitState init;
  init.P.setIdentity();
  const SimConfig& c = run.config;
  filter::FilterState s(init, c.R_IC, c.p_IC, c.time_offset, options, Eigen::Matrix<double, 7, 7>::Identity());
  for (size_t k : w.epochs) {
    s.nav = run.epochs[k].*run.variant.lin;
    s.addClone(run.epochs[k].t, Eigen::Vector3d::Zero());
  }
  return s;
}

// Fixed estimates of the window's landmarks, each anchored in the camera of
// its first observing image (at that image's linearization point, a clone of s).
std::vector<AnchoredPoint> landmarkEstimates(const SimData& d, const Window& w, const filter::FilterState& s) {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> n(0.0, kLandmarkSigma);
  const Sophus::SE3d T_IC(s.R_IC, s.p_IC);
  std::vector<AnchoredPoint> out;
  for (const WindowLandmark& l : w.landmarks) {
    const filter::Clone& c = s.clones()[l.seen.front()];
    const Eigen::Vector3d p = d.landmarks[static_cast<size_t>(l.id)].p_G + Eigen::Vector3d(n(rng), n(rng), n(rng));
    out.push_back(AnchoredPoint::fromPoint(Sophus::SE3d(c.R_GI, c.p_GI) * T_IC, p));
  }
  return out;
}

WindowModel windowModel(const FilterRun& run, const Window& w) {
  WindowModel m{cloneWindow(run, w), {}, {Matrix15d::Identity()}};
  m.landmarks = landmarkEstimates(run.data, w, m.state);
  for (size_t j = 1; j < w.epochs.size(); ++j) m.Phi.push_back(run.epochs[w.epochs[j]].Phi * m.Phi.back());
  return m;
}

// Rows of O for landmark i: per image that sees it, [H_imu Phi(k, 1) | H_f]
// (2 rows each, 15 + 3 columns). Empty if the landmark cannot be linearized.
std::optional<Eigen::MatrixXd> landmarkRows(const CameraModel& camera, const WindowModel& m,
                                            const WindowLandmark& l, size_t i) {
  filter::FeatureTrack track;
  track.uv = l.uv;
  filter::FeatureJacobian fj;
  if (!filter::linearizeFeature(m.state, camera, track, l.seen, m.landmarks[i], &fj)) return std::nullopt;
  Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(2 * static_cast<long>(l.seen.size()), 18);
  for (size_t j = 0; j < l.seen.size(); ++j) {
    const long r = 2 * static_cast<long>(j);
    Eigen::Matrix<double, 2, 15> H_imu = Eigen::Matrix<double, 2, 15>::Zero();
    H_imu.leftCols<6>() = fj.H_x.block(r, m.state.cloneIndex(l.seen[j]), 2, 6);
    rows.block(r, 0, 2, 15) = H_imu * m.Phi[l.seen[j]];
    rows.block(r, 15, 2, 3) = fj.H_f.middleRows(r, 2);
  }
  return rows;
}

// O over the IMU state at the first image and the landmarks; empty if a
// landmark cannot be linearized.
std::optional<Eigen::MatrixXd> observabilityMatrix(const FilterRun& run, const Window& w, const WindowModel& m) {
  std::vector<Eigen::MatrixXd> blocks;
  long rows = 0;
  for (size_t i = 0; i < w.landmarks.size(); ++i) {
    std::optional<Eigen::MatrixXd> b = landmarkRows(run.camera, m, w.landmarks[i], i);
    if (!b) return std::nullopt;
    rows += b->rows();
    blocks.push_back(std::move(*b));
  }
  Eigen::MatrixXd O = Eigen::MatrixXd::Zero(rows, 15 + 3 * static_cast<long>(w.landmarks.size()));
  long row = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const long n = blocks[i].rows();
    O.block(row, 0, n, 15) = blocks[i].leftCols<15>();
    O.block(row, 15 + 3 * static_cast<long>(i), n, 3) = blocks[i].rightCols<3>();
    row += n;
  }
  return O;
}

WindowResult analyze(const Eigen::MatrixXd& O, const Eigen::MatrixXd& N) {
  const Eigen::VectorXd scale = O.colwise().norm().cwiseMax(1e-300).cwiseInverse().transpose();
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(O * scale.asDiagonal());
  const Eigen::VectorXd sv = svd.singularValues() / svd.singularValues()(0);
  WindowResult r;
  r.smallest = sv.tail(6).reverse();
  r.unobservable = static_cast<int>(O.cols() - (sv.array() >= kRankThreshold).count());
  for (int j = 0; j < 4; ++j) r.on(j) = (O * N.col(j)).norm() / (O.norm() * N.col(j).norm());
  return r;
}

std::optional<WindowResult> analyzeWindow(const FilterRun& run, const Window& w) {
  const WindowModel m = windowModel(run, w);
  const std::optional<Eigen::MatrixXd> O = observabilityMatrix(run, w, m);
  if (!O) return std::nullopt;
  return analyze(*O, filter::unobservableDirections(run.epochs[w.epochs.front()].*run.variant.lin, m.landmarks));
}

void writeRow(std::ostream& csv, const FilterRun& run, const Window& w, const WindowResult& r) {
  csv << run.variant.fej << ',' << run.epochs[w.epochs.front()].t << ','
      << w.landmarks.size() << ',' << r.unobservable;
  for (int j = 0; j < 6; ++j) csv << ',' << r.smallest(j);
  for (int j = 0; j < 4; ++j) csv << ',' << r.on(j);
  csv << '\n';
}

// One row per window of the run; windows with fewer than 3 landmarks or that
// cannot be linearized are logged and skipped.
void analyzeRun(const FilterRun& run, const Args& a, std::ostream& csv) {
  const size_t step = static_cast<size_t>(a.stride * run.config.cam_rate);
  for (size_t start = step; start + static_cast<size_t>(a.window) <= run.epochs.size(); start += step) {
    const Window w = makeWindow(run, start, a);
    const std::optional<WindowResult> r =
        w.landmarks.size() < 3 ? std::nullopt : analyzeWindow(run, w);
    if (r) writeRow(csv, run, w, *r);
    else std::cout << "skipped window at t = " << run.epochs[start].t << " s\n";
  }
}

std::vector<filter::SimEpoch> runFilter(const Simulator& sim, const SimData& d, const Variant& v) {
  filter::MsckfOptions options;
  options.state.fej = v.fej;
  options.update.pixel_noise = sim.config().pixel_noise;
  filter::SimRunConfig rc;
  rc.seed = 1000 + sim.config().seed;
  return filter::runFilterOnSimulation(sim, simulatedCamera(), d, options, rc);
}

int run(const Args& a) {
  const Scenario scenario = parseScenario(a.scenario);
  SimConfig c = defaultSimConfig(scenario);
  c.duration = a.duration + 2.0;
  c.seed = static_cast<uint64_t>(a.seed);
  const CameraModelPtr camera = simulatedCamera();
  const Simulator sim(c, makeScenario(scenario, c.duration + 1.0, 1), camera);
  const SimData d = sim.run();
  fs::create_directories(a.out);
  std::ofstream csv(fs::path(a.out) / "windows.csv");
  csv << "fej,t0,landmarks,unobservable,sv1,sv2,sv3,sv4,sv5,sv6,on_tx,on_ty,on_tz,on_yaw\n" << std::setprecision(6);
  for (const Variant& v : kVariants) analyzeRun({c, d, *camera, runFilter(sim, d, v), v}, a, csv);
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
    else if (arg == "--seed") a.seed = std::stoi(next());
    else if (arg == "--window") a.window = std::stoi(next());
    else if (arg == "--stride") a.stride = std::stod(next());
    else if (arg == "--features") a.features = std::stoi(next());
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
    std::cerr << "msckf_observability: " << e.what() << "\n";
    return 2;
  }
}
