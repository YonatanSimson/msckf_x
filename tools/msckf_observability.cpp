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
// with each Phi from imuTransition and each H from linearizeFeature, evaluated
// where the filter evaluates them: the propagated estimate (FEJ) or the updated
// one (no FEJ), the filter's biases, a fixed estimate of each landmark. N is
// the analytic nullspace of the system at the first image: global translation
// and yaw about gravity (body-frame dtheta = R^T z, dp = z x p, dv = z x v, a
// landmark moved by z x p_f). With FEJ, O N = 0 and O loses exactly those 4
// directions; without FEJ, yaw gains spurious information.
//
// Output: windows.csv, one row per window and FEJ setting: the 6 smallest
// singular values of O (columns scaled to unit norm) over the largest, the
// number below 1e-8 (unobservable directions), and |O n| / (|O| |n|) for each
// nullspace direction.

#include "msckf/camera/pinhole_radtan.h"
#include "msckf/filter/feature_update.h"
#include "msckf/filter/observability.h"
#include "msckf/filter/propagator.h"
#include "msckf/filter/sim_run.h"
#include "msckf/sim/scenarios.h"

#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <string>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

struct Args {
  std::string scenario = "handheld", out = "observability_out";
  double duration = 30.0, stride = 5.0;
  int seed = 1, window = 20, features = 20;
};

struct WindowLandmark {
  int64_t id = 0;
  std::vector<size_t> seen;  // window images (0 .. K-1) that observe it
  std::vector<Eigen::Vector2d> uv;
};

struct Window {
  std::vector<size_t> epochs;  // into the run's epochs
  std::vector<WindowLandmark> landmarks;
};

constexpr double kLandmarkSigma = 0.05;  // m, error of each landmark's fixed estimate
constexpr double kRankThreshold = 1e-8;  // singular value / largest

NavState linearizationPoint(const filter::SimEpoch& e, bool fej) { return fej ? e.nav_prior : e.nav_post; }

Scenario parseScenario(const std::string& s) {
  if (s == "handheld") return Scenario::kHandheld;
  if (s == "cart") return Scenario::kCart;
  if (s == "fast_rotation") return Scenario::kFastRotation;
  if (s == "car") return Scenario::kCar;
  throw std::invalid_argument("unknown scenario " + s);
}

// The filter's IMU transition from epoch a to epoch b.
Matrix15d transition(const SimData& d, const filter::SimEpoch& a, const filter::SimEpoch& b, bool fej) {
  ImuPreintegration pre(a.gyro_bias, a.acc_bias, ImuNoise{});
  pre.integrateRange(d.imu, a.t_state, b.t_state);
  return filter::imuTransition(pre, linearizationPoint(a, fej), b.nav_prior, ImuNoise{}).Phi;
}

// Images k = start .. start + K - 1 and up to L landmarks seen in at least 3
// of them (the most observed first), as the MSCKF's partial tracks.
Window makeWindow(const SimData& d, const std::vector<filter::SimEpoch>& epochs, size_t start, const Args& a) {
  Window w;
  std::map<int64_t, WindowLandmark> by_id;
  for (size_t k = start; k < start + static_cast<size_t>(a.window) && k < epochs.size(); ++k) {
    size_t f = 0;
    while (f < d.frames.size() && d.frames[f].t_cam < epochs[k].t - 1e-9) ++f;
    for (const Observation& o : d.frames[f].observations) {
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

// A filter state whose clones are the window's IMU linearization points, so
// that linearizeFeature evaluates each image's H there.
filter::FilterState cloneWindow(const SimConfig& c, const std::vector<filter::SimEpoch>& epochs, const Window& w,
                                bool fej) {
  filter::StateOptions options;
  options.fej = false;  // the clones already hold the chosen linearization points
  options.estimate_extrinsic = false;
  options.estimate_time_offset = false;
  ImuInitState init;
  init.P.setIdentity();
  filter::FilterState s(init, c.R_IC, c.p_IC, c.time_offset, options, Eigen::Matrix<double, 7, 7>::Identity());
  for (size_t k : w.epochs) {
    s.nav = linearizationPoint(epochs[k], fej);
    s.addClone(epochs[k].t, Eigen::Vector3d::Zero());
  }
  return s;
}

// Rows of O for one landmark: per image that sees it, [H_imu Phi(k, 1), H_f in
// the landmark's columns].
bool addLandmarkRows(const filter::FilterState& s, const CameraModel& cam, const WindowLandmark& l, size_t col,
                     const AnchoredPoint& f, const std::vector<Matrix15d>& Phi, Eigen::MatrixXd* O, long* row) {
  filter::FeatureTrack track;
  track.uv = l.uv;
  filter::FeatureJacobian fj;
  if (!filter::linearizeFeature(s, cam, track, l.seen, f, &fj)) return false;
  for (size_t j = 0; j < l.seen.size(); ++j) {
    const long r = 2 * static_cast<long>(j);
    Eigen::Matrix<double, 2, 15> H_imu = Eigen::Matrix<double, 2, 15>::Zero();
    H_imu.leftCols<6>() = fj.H_x.block(r, s.cloneIndex(l.seen[j]), 2, 6);
    O->block(*row, 0, 2, 15) = H_imu * Phi[l.seen[j]];
    O->block(*row, 15 + 3 * static_cast<long>(col), 2, 3) = fj.H_f.middleRows(r, 2);
    *row += 2;
  }
  return true;
}

struct WindowResult {
  Eigen::VectorXd smallest;  // 6 smallest singular values / largest, columns scaled
  int unobservable = 0;
  Eigen::Vector4d on = Eigen::Vector4d::Zero();  // |O n| / (|O| |n|) per nullspace direction
};

WindowResult analyze(const Eigen::MatrixXd& O, const Eigen::MatrixXd& N) {
  const Eigen::VectorXd scale = O.colwise().norm().cwiseMax(1e-300).cwiseInverse().transpose();
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(O * scale.asDiagonal());
  const Eigen::VectorXd sv = svd.singularValues() / svd.singularValues()(0);
  WindowResult r;
  r.smallest = sv.tail(6).reverse();
  r.unobservable = static_cast<int>((sv.array() < kRankThreshold).count());
  const double norm_O = Eigen::JacobiSVD<Eigen::MatrixXd>(O).singularValues()(0);
  for (int j = 0; j < 4; ++j) r.on(j) = (O * N.col(j)).norm() / (norm_O * N.col(j).norm());
  return r;
}

bool analyzeWindow(const SimConfig& c, const CameraModel& cam, const SimData& d,
                   const std::vector<filter::SimEpoch>& epochs, const Window& w, bool fej, WindowResult* out) {
  const NavState x1 = linearizationPoint(epochs[w.epochs.front()], fej);
  const filter::FilterState s = cloneWindow(c, epochs, w, fej);
  const std::vector<AnchoredPoint> f = landmarkEstimates(d, w, s);
  std::vector<Matrix15d> Phi{Matrix15d::Identity()};
  for (size_t j = 1; j < w.epochs.size(); ++j) {
    Phi.push_back(transition(d, epochs[w.epochs[j - 1]], epochs[w.epochs[j]], fej) * Phi.back());
  }
  long rows = 0;
  for (const WindowLandmark& l : w.landmarks) rows += 2 * static_cast<long>(l.seen.size());
  Eigen::MatrixXd O = Eigen::MatrixXd::Zero(rows, 15 + 3 * static_cast<long>(w.landmarks.size()));
  long row = 0;
  for (size_t l = 0; l < w.landmarks.size(); ++l) {
    if (!addLandmarkRows(s, cam, w.landmarks[l], l, f[l], Phi, &O, &row)) return false;
  }
  *out = analyze(O, filter::unobservableDirections(x1, f));
  return true;
}

int run(const Args& a) {
  Eigen::VectorXd k(9);  // EuRoC cam0, as msckf_sim_eval
  k << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0;
  const auto camera = std::make_shared<PinholeRadtan>(752, 480, k);
  const Scenario scenario = parseScenario(a.scenario);
  SimConfig c = defaultSimConfig(scenario);
  c.duration = a.duration + 2.0;
  c.seed = static_cast<uint64_t>(a.seed);
  const Simulator sim(c, makeScenario(scenario, c.duration + 1.0, 1), camera);
  const SimData d = sim.run();

  fs::create_directories(a.out);
  std::ofstream csv(fs::path(a.out) / "windows.csv");
  csv << "fej,t0,landmarks,unobservable,sv1,sv2,sv3,sv4,sv5,sv6,on_tx,on_ty,on_tz,on_yaw\n" << std::setprecision(6);
  for (const bool fej : {true, false}) {
    filter::MsckfOptions options;
    options.state.fej = fej;
    options.update.pixel_noise = c.pixel_noise;
    filter::SimRunConfig rc;
    rc.seed = static_cast<uint64_t>(1000 + a.seed);
    const auto epochs = filter::runFilterOnSimulation(sim, camera, d, options, rc);
    const size_t step = static_cast<size_t>(a.stride * c.cam_rate);
    for (size_t start = step; start + static_cast<size_t>(a.window) <= epochs.size(); start += step) {
      const Window w = makeWindow(d, epochs, start, a);
      WindowResult r;
      if (w.landmarks.size() < 3 || !analyzeWindow(c, *camera, d, epochs, w, fej, &r)) continue;
      csv << fej << ',' << epochs[start].t << ',' << w.landmarks.size() << ',' << r.unobservable;
      for (int j = 0; j < 6; ++j) csv << ',' << r.smallest(j);
      for (int j = 0; j < 4; ++j) csv << ',' << r.on(j);
      csv << '\n';
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
      else if (arg == "--duration") a.duration = std::stod(next());
      else if (arg == "--seed") a.seed = std::stoi(next());
      else if (arg == "--window") a.window = std::stoi(next());
      else if (arg == "--stride") a.stride = std::stod(next());
      else if (arg == "--features") a.features = std::stoi(next());
      else if (arg == "--out") a.out = next();
      else throw std::invalid_argument("unknown option " + arg);
    }
    return run(a);
  } catch (const std::exception& e) {
    std::cerr << "msckf_observability: " << e.what() << "\n";
    return 2;
  }
}
