// Stage 1b evaluation (plan, "Stage 1 metrics"): runs the feature-less
// dynamic initialization at the start of every segment of a sequence, with
// several window lengths, and writes one row per run with its errors against
// ground truth. The gate checker (python/msckf_gates/stage1b_dynamic_init.py)
// computes the metrics from runs.csv.
//
//   dynamic_init_eval euroc --euroc DIR --cam camN [--camchain YAML] --gt GT.csv --tracks TRACKS.csv
//                           [--imu-yaml YAML] [--bias gt|zero] --out DIR
//   dynamic_init_eval sim --scenario handheld|cart|fast_rotation [--seeds N] --out DIR
//
// --refine always|fallback|never: the retry pipeline's refinement policy
// (InitRefinementPolicy); --linear-cov-scale: variance factor of the linear
// hand-off. Each window also gets an "nr" row: the paper's no-refinement
// variant (linear solution + linearHandoff covariance, scale gate).
// --restart S: after a failed retry, start the pipeline again every S s
// until it succeeds ("restart" rows: window = time from the start to init).
// euroc also writes trace.csv (TraceLog): per frame of every pipeline run,
// estimated and true gravity in the camera frame and speed, for
// python/play_init.py.
//
// euroc (EuRoC and TUM-VI): bearings come from the tracks of the front end
// (tracker_eval tracks: frame, t_ns, track, u, v), unprojected through the
// sequence's camera model; IMU from mav0/imu0/data.csv. Ground truth GT.csv
// is either EuRoC's state_groundtruth_estimate0 (pose, velocity, biases) or a
// pose-only file (TUM-VI dso/gt_imu.csv). Velocity truth: the state file's
// velocity when it has one (EuRoC; KITTI as converted, the OXTS's fused
// velocity), else the central difference of the interpolated positions
// (+-10 ms). --bias gt (default) gives the initializer EuRoC's
// ground-truth biases as the "prior calibration" the paper assumes; zero uses
// none (TUM-VI's IMU data is already calibrated, and has no bias truth).
// --imu-yaml: Kalibr-style noise densities (default: ImuNoise defaults);
// they weight the IMU in the refinement, scaled by --imu-noise-scale (real-data
// inflation, default 1). Bearing noise: --pixel-sigma px (default 1) at the
// image centre.
//
// sim: the simulator's scenarios, with --seeds seeds (default 10), observation
// bearings unprojected from the noisy pixels, biases known.
//
// Common options: --segment S (10), --windows 0.1,0.2,0.3,0.5,0.75,1.0.
//
// Each window is reduced to its keyframes (selectKeyframes: by parallax, at
// most 5, as in the paper) before both steps.
//
// runs.csv, three rows per window, one per method:
//   linear   the feature-less solve (step 1)
//   refined  step 1 + the refinement (step 2): the pipeline
//   okvis    OKVIS2-X's start, as a baseline: gravity from the mean
//            accelerometer reading over the window length before the first
//            frame (ImuError::initPose), velocity 0 (its speed prior)
// Columns: sequence, t0, window, frames (keyframes used), method, ok, reason, singular_value,
// acc_excitation, pairs, iterations, gravity_err_deg, velocity_err,
// velocity_norm, scale_err_pct, time_ms. Gravity and velocity are compared in
// the first frame's IMU frame (independent of the yaw gauge). Scale: camera
// translations between frame pairs (linear: the keyframe pairs; refined: all
// pairs of clones) against ground truth, as the least-squares ratio r;
// 100 (max(r, 1/r) - 1). okvis has no scale.

#include "msckf/calib/kalibr_io.h"
#include "msckf/core/constants.h"
#include "msckf/eval/trajectory.h"
#include "msckf/imu/preintegration.h"
#include "msckf/init/dynamic_init.h"
#include "msckf/init/featureless_init.h"
#include "msckf/init/init_refinement.h"
#include "msckf/io/euroc.h"
#include "msckf/sim/scenarios.h"
#include "msckf/sim/simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace msckf;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void usage() {
  std::cerr << "usage: dynamic_init_eval euroc --euroc DIR --cam camN [--camchain YAML] --gt GT.csv "
               "--tracks TRACKS.csv [--bias gt|zero] --out DIR\n"
               "       dynamic_init_eval sim --scenario handheld|cart|fast_rotation [--seeds N] --out DIR\n"
               "       common: [--segment S] [--windows 0.1,0.2,...]\n";
}

struct Args {
  std::string mode, euroc, cam = "cam0", camchain, gt, tracks, imu_yaml, bias = "gt", scenario = "handheld", out;
  double segment = 10.0;
  std::vector<double> windows{0.1, 0.2, 0.3, 0.5, 0.75, 1.0};
  int seeds = 10;
  double imu_noise_scale = 1.0;  // multiplies the IMU noise densities (real-data inflation)
  double pixel_sigma = 1.0;      // px, bearing noise at the image centre
  InitRefinementPolicy refine = InitRefinementPolicy::kAlways;  // retry rows (--refine always|fallback|never)
  double linear_covariance_scale = DynamicInitConfig{}.linear_covariance_scale;  // --linear-cov-scale
  bool triplets = FeaturelessInitConfig{}.triplet_scale;  // --triplets: add the relative-scale equations
  double triplet_weight = FeaturelessInitConfig{}.triplet_weight;               // --triplet-weight
  double triplet_parallax = FeaturelessInitConfig{}.triplet_min_parallax_deg;   // --triplet-parallax (deg)
  double sim_pixel_noise = -1.0;  // sim: override SimConfig::pixel_noise (px) when >= 0 (--sim-pixel-noise)
  double still_noise_px = 0.0;  // sim: its pixel noise (see evaluateRetry)
  double restart = 0.0;  // s; > 0: after a failed retry, start again every `restart` s ("restart" rows)
  double min_excitation = DynamicInitConfig{}.linear.min_acc_excitation;  // m/s^2 (the pipeline's: 0, scale gate instead)
};

// Ground truth for one window: the first IMU frame's gravity and velocity,
// and every frame's camera position (for the scale).
struct Truth {
  Eigen::Vector3d g_I0, v_I0;
  std::vector<Eigen::Vector3d> p_C;  // per frame, world frame
  Eigen::Vector3d v_body_last = Eigen::Vector3d::Zero();  // IMU-frame velocity at the last frame
};

class RunLog {
 public:
  explicit RunLog(const fs::path& path) : f_(path) {
    f_ << "sequence,t0,window,frames,method,ok,reason,singular_value,acc_excitation,pairs,iterations,"
          "gravity_err_deg,velocity_err,velocity_norm,scale_err_pct,time_ms,velocity_sigma,tilt_sigma_deg,velocity_nees,speed_est,init\n"
       << std::setprecision(9);
  }

  struct Row {
    std::string method, reason = "ok";
    bool ok = false, solved = false;
    double singular_value = kNaN, acc_excitation = kNaN, ms = 0.0, scale_err = kNaN;
    double velocity_sigma = kNaN, tilt_sigma_deg = kNaN;  // refined: from the hand-off covariance
    double velocity_nees = kNaN;  // refined: of the IMU-frame velocity at the hand-off (3 dof, yaw free)
    int pairs = 0, iterations = 0;
    std::string init;  // retry rows: "static" or "dynamic" (the hand-off's path)
    Eigen::Vector3d g_I0 = Eigen::Vector3d::Zero(), v_I0 = Eigen::Vector3d::Zero();
  };

  void add(const std::string& seq, double t0, double window, size_t frames, const Row& r, const Truth& truth) {
    const double g_err = std::atan2(r.g_I0.cross(truth.g_I0).norm(), r.g_I0.dot(truth.g_I0)) * 180.0 / M_PI;
    f_ << seq << ',' << t0 << ',' << window << ',' << frames << ',' << r.method << ',' << r.ok << ','
       << (r.ok ? "ok" : r.reason) << ',' << r.singular_value << ',' << r.acc_excitation << ',' << r.pairs << ','
       << r.iterations << ',' << (r.solved ? g_err : kNaN) << ',' << (r.solved ? (r.v_I0 - truth.v_I0).norm() : kNaN)
       << ',' << truth.v_I0.norm() << ',' << (r.solved ? r.scale_err : kNaN) << ',' << r.ms << ','
       << r.velocity_sigma << ',' << r.tilt_sigma_deg << ',' << r.velocity_nees << ','
       << (r.solved ? r.v_I0.norm() : kNaN) << ',' << r.init << '\n';
  }

 private:
  std::ofstream f_;
};

// 100 (max(r, 1/r) - 1) for the least-squares ratio r of estimated to true
// camera translations.
double scaleError(const std::vector<double>& est, const std::vector<double>& truth) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < est.size(); ++i) {
    num += est[i] * truth[i];
    den += truth[i] * truth[i];
  }
  const double r = den > 0.0 ? num / den : kNaN;
  return r > 0.0 ? 100.0 * (std::max(r, 1.0 / r) - 1.0) : kNaN;
}

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Runs the three methods on one window and logs a row for each:
//  linear   the feature-less solve (step 1)
//  refined  step 1 + the refinement (step 2): our pipeline
//  okvis    OKVIS2-X's start (ImuError::initPose and its speed prior):
//           gravity from the mean accelerometer reading over the IMU samples
//           of the window length before the first frame, v = 0.
// Ground truth for a set of frame times (the first is {I0}); nullopt where
// the dataset's ground truth does not cover them.
using TruthFn = std::function<std::optional<Truth>(const std::vector<double>&)>;

// Velocity NEES of the IMU-frame velocity at the hand-off (3 dof, yaw free):
// v_b = R^T v, d v_b = [R^T v]x dtheta + R^T dv.
double bodyVelocityNees(const ImuInitState& st, const Eigen::Vector3d& v_body_true) {
  const Eigen::Matrix3d Rt = st.nav.R_GI.inverse().matrix();
  const Eigen::Vector3d vb = Rt * st.nav.v_GI;
  Eigen::Matrix<double, 3, 15> J = Eigen::Matrix<double, 3, 15>::Zero();
  J.block<3, 3>(0, imu_err::kTheta) = Sophus::SO3d::hat(vb);
  J.block<3, 3>(0, imu_err::kV) = Rt;
  const Eigen::Matrix3d Pvb = J * st.P * J.transpose();
  const Eigen::Vector3d e = v_body_true - vb;
  return e.dot(Pvb.ldlt().solve(e));
}

// Hand-off sigmas: largest velocity sigma, and the tilt sigma (horizontal
// components of the global-frame rotation error).
void handoffSigmas(const ImuInitState& st, RunLog::Row& r) {
  const Eigen::Matrix3d Pv = st.P.block<3, 3>(imu_err::kV, imu_err::kV);
  r.velocity_sigma = std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Pv).eigenvalues().maxCoeff());
  const Eigen::Matrix<double, 2, 3> H = st.nav.R_GI.matrix().topRows<2>();
  const Eigen::Matrix2d Pt = H * st.P.block<3, 3>(imu_err::kTheta, imu_err::kTheta) * H.transpose();
  r.tilt_sigma_deg = std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(Pt).eigenvalues().maxCoeff()) * 180.0 / M_PI;
}

// Static hand-off row (truth at the hand-off time): gravity and velocity in
// that IMU frame; no scale.
RunLog::Row staticRow(const ImuInitState& st, const Truth& truth) {
  RunLog::Row r;
  r.ok = r.solved = true;
  r.g_I0 = st.nav.R_GI.inverse() * kGravity;
  r.v_I0 = st.nav.R_GI.inverse() * st.nav.v_GI;
  handoffSigmas(st, r);
  r.velocity_nees = bodyVelocityNees(st, truth.v_body_last);
  return r;
}

// Linear hand-off row (the paper's NR): gravity and velocity of the linear
// solution at the first frame, its pair scales, and sigmas, NEES and the
// scale gate from linearHandoff's covariance at the last keyframe.
RunLog::Row linearHandoffRow(const FeaturelessInitResult& lin, const ImuInitState& st, const Truth& truth) {
  RunLog::Row r;
  r.ok = r.solved = true;
  r.g_I0 = lin.g_I0;
  r.v_I0 = lin.v_I0;
  std::vector<double> est, tru;
  for (size_t i = 0; i < lin.pair_scale.size(); ++i) {
    const auto [j, k] = lin.pair_frames[i];
    est.push_back(lin.pair_scale[i]);
    tru.push_back((truth.p_C[k] - truth.p_C[j]).norm());
  }
  r.scale_err = scaleError(est, tru);
  handoffSigmas(st, r);
  r.velocity_nees = bodyVelocityNees(st, truth.v_body_last);
  return r;
}

// Refined-solution row: gravity and velocity of the first clone, scale of the
// clones' camera translations over all pairs.
RunLog::Row refinedRow(const InitRefinementResult& ref, const Sophus::SE3d& T_IC, const Truth& truth) {
  RunLog::Row r;
  r.ok = r.solved = true;
  r.iterations = ref.iterations;
  const NavState& c0 = ref.clones.front();
  r.g_I0 = c0.R_GI.inverse() * kGravity;
  r.v_I0 = c0.R_GI.inverse() * c0.v_GI;
  std::vector<double> est, tru;
  for (size_t j = 0; j < ref.clones.size(); ++j) {
    for (size_t k = j + 1; k < ref.clones.size(); ++k) {
      const NavState &a = ref.clones[j], &b = ref.clones[k];
      est.push_back(((b.p_GI + b.R_GI * T_IC.translation()) - (a.p_GI + a.R_GI * T_IC.translation())).norm());
      tru.push_back((truth.p_C[k] - truth.p_C[j]).norm());
    }
  }
  r.scale_err = scaleError(est, tru);
  handoffSigmas(ref.state, r);
  r.velocity_nees = bodyVelocityNees(ref.state, truth.v_body_last);
  return r;
}

void evaluateWindow(RunLog& log, const std::string& seq, double t0, double window,
                    const std::vector<ImuSample>& imu, const std::vector<InitFrame>& all_frames,
                    const Sophus::SE3d& T_IC, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                    const ImuNoise& noise, double bearing_sigma, const TruthFn& truth_for,
                    const FeaturelessInitConfig& lin_cfg, double linear_covariance_scale) {
  using Row = RunLog::Row;
  auto start = std::chrono::steady_clock::now();
  // Keyframes (the paper's: by parallax, at most 5), for both steps.
  std::vector<InitFrame> win;
  std::vector<double> times;
  for (int k : selectKeyframes(imu, all_frames, T_IC, bg)) {
    win.push_back(all_frames[k]);
    times.push_back(all_frames[k].t);
  }
  const std::optional<Truth> maybe_truth = truth_for(times);
  if (!maybe_truth) return;
  const Truth& truth = *maybe_truth;
  const FeaturelessInitResult lin = featurelessInit(imu, win, T_IC, bg, ba, noise, lin_cfg);
  Row l;
  l.method = "linear";
  l.ms = msSince(start);
  l.ok = lin.ok;
  l.reason = lin.reason;
  l.solved = lin.pairs > 0 && lin.reason != "pairs" && lin.reason != "imu";
  l.singular_value = lin.singular_value;
  l.acc_excitation = lin.acc_excitation;
  l.pairs = lin.pairs;
  l.g_I0 = lin.g_I0;
  l.v_I0 = lin.v_I0;
  {
    std::vector<double> est, tru;
    for (size_t i = 0; i < lin.pair_scale.size(); ++i) {  // empty unless solved
      const auto [j, k] = lin.pair_frames[i];
      est.push_back(lin.pair_scale[i]);
      tru.push_back((truth.p_C[k] - truth.p_C[j]).norm());
    }
    l.scale_err = scaleError(est, tru);
  }
  log.add(seq, t0, window, win.size(), l, truth);

  Row r = l;
  r.method = "refined";
  r.solved = false;
  if (lin.ok) {
    InitRefinementConfig rc;
    rc.bearing_sigma = bearing_sigma;
    const InitRefinementResult ref = refineInit(imu, win, T_IC, lin, bg, ba, noise, rc);
    const double ms = msSince(start);
    if (ref.ok) {
      // Logged in full even when the scale gate refuses it (ok = 0, reason
      // "scale"), so the gate itself can be evaluated against the truth.
      const Row base = r;
      r = refinedRow(ref, T_IC, truth);
      r.method = base.method;
      r.singular_value = base.singular_value;
      r.acc_excitation = base.acc_excitation;
      r.pairs = base.pairs;
      if (!passesScaleGate(ref.state, DynamicInitConfig{})) {
        r.ok = false;
        r.reason = "scale";
      }
    } else {
      r.ok = false;
      r.reason = ref.reason;
      r.iterations = ref.iterations;
    }
    r.ms = ms;
  }
  log.add(seq, t0, window, win.size(), r, truth);

  Row nr = l;
  nr.method = "nr";
  nr.solved = false;
  if (lin.ok) {
    DynamicInitConfig dc;
    dc.linear_covariance_scale = linear_covariance_scale;
    dc.linear = lin_cfg;
    if (const auto st = linearHandoff(imu, win, T_IC, lin, bg, ba, noise, dc)) {
      const Row base = nr;
      nr = linearHandoffRow(lin, *st, truth);
      nr.method = base.method;
      nr.singular_value = base.singular_value;
      nr.acc_excitation = base.acc_excitation;
      nr.pairs = base.pairs;
      nr.ms = base.ms;
      if (!passesScaleGate(*st, dc)) {
        nr.ok = false;
        nr.reason = "scale";
      }
    } else {
      nr.ok = false;
      nr.reason = "covariance";
    }
  }
  log.add(seq, t0, window, win.size(), nr, truth);

  Row o;
  o.method = "okvis";
  start = std::chrono::steady_clock::now();
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  int count = 0;
  for (const ImuSample& smp : imu) {
    if (smp.t >= win.front().t - window && smp.t <= win.front().t) {
      mean += smp.acc - ba;
      ++count;
    }
  }
  o.ms = msSince(start);
  o.ok = o.solved = count > 0;
  o.reason = count > 0 ? "ok" : "imu";
  o.g_I0 = -kGravity.norm() * mean.normalized();
  log.add(seq, t0, window, win.size(), o, truth);
}

// Per-frame truth for the trace: gravity direction in the camera frame and
// the speed; nullopt where the ground truth does not cover t.
struct FrameTruth {
  Eigen::Vector3d g_C;
  double speed;
};
using FrameTruthFn = std::function<std::optional<FrameTruth>(double)>;

// trace.csv: for each pipeline run, every frame from its start to the end of
// its last window, with the estimated and true gravity direction in the
// camera frame and the estimated and true speed (play_init.py draws them).
// The estimate is the hand-off's (static, or the first refined clone), or for
// a refused attempt the linear solution when it has one, carried to every
// frame by IMU preintegration from that anchor.
class TraceLog {
 public:
  explicit TraceLog(const fs::path& path) : f_(path) {
    f_ << "run,t0,attempt_start,method,ok,init,reason,window,t,gx_est,gy_est,gz_est,gx_true,gy_true,gz_true,"
          "speed_est,speed_true,features\n"
       << std::setprecision(12);
  }

  void add(const std::string& method, double t0_label, double t_start, const DynamicInitResult& res,
           const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames, const Sophus::SE3d& T_IC,
           const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, const ImuNoise& noise, const FrameTruthFn& truth) {
    // Anchor: a state at a time, with its biases.
    std::optional<NavState> anchor;
    double t_a = 0.0;
    Eigen::Vector3d abg = bg, aba = ba;
    if ((res.ok || res.reason == "scale") && res.is_static) {
      anchor = res.refinement.state.nav;
      t_a = res.refinement.state.t;
      abg = res.refinement.state.gyro_bias;
      aba = res.refinement.state.acc_bias;
    } else if ((res.ok || res.reason == "scale") && !res.refinement.clones.empty()) {
      anchor = res.refinement.clones.front();
      t_a = res.keyframe_t.front();
      abg = res.refinement.gyro_bias;
      aba = res.refinement.acc_bias;
    } else if (res.linear.ok && !res.keyframe_t.empty()) {
      NavState s;
      s.R_GI = attitudeFromUp(-res.linear.g_I0.normalized());
      s.v_GI = s.R_GI * res.linear.v_I0;
      anchor = s;
      t_a = res.keyframe_t.front();
    }
    const double t_end = res.keyframe_t.empty() ? t_start + res.window : res.keyframe_t.back();
    const std::string init =
        res.ok ? (res.in_place ? "in_place" : res.is_static ? "static" : res.linear_only ? "linear" : "dynamic") : "";
    for (const InitFrame& fr : frames) {
      if (fr.t < t_start - 1e-3) continue;
      if (fr.t > std::max(t_end, t_start + res.window) + 1e-3) break;
      Eigen::Vector3d g_est = Eigen::Vector3d::Constant(kNaN);
      double v_est = kNaN;
      if (anchor) {
        NavState s = *anchor;
        if (std::abs(fr.t - t_a) > 1e-6) {
          ImuPreintegration pre(abg, aba, noise);
          if (fr.t > t_a) {
            pre.integrateRange(imu, t_a, fr.t);
            s = pre.predict(*anchor, abg, aba);
          } else {
            // Backwards: anchor = predict(s), so R = R_a dR^T and
            // v = v_a - g T - R dv.
            pre.integrateRange(imu, fr.t, t_a);
            s.R_GI = anchor->R_GI * pre.deltaR().inverse();
            s.v_GI = anchor->v_GI - kGravity * pre.deltaTime() - s.R_GI * pre.deltaV();
          }
        }
        g_est = ((s.R_GI * T_IC.so3()).inverse() * kGravity).normalized();
        v_est = s.v_GI.norm();
      }
      const std::optional<FrameTruth> tr = truth(fr.t);
      const Eigen::Vector3d g_true = tr ? tr->g_C : Eigen::Vector3d::Constant(kNaN);
      f_ << run_ << ',' << t0_label << ',' << t_start << ',' << method << ',' << res.ok << ',' << init << ','
         << (res.ok ? "ok" : res.reason) << ',' << res.window << ',' << fr.t << ',' << g_est.x() << ','
         << g_est.y() << ',' << g_est.z() << ',' << g_true.x() << ',' << g_true.y() << ',' << g_true.z() << ','
         << v_est << ',' << (tr ? tr->speed : kNaN) << ',' << fr.bearings.size() << '\n';
    }
    ++run_;
  }

 private:
  std::ofstream f_;
  int run_ = 0;
};

// The pipeline with retry (initializeDynamic) from t0: one row, method
// "retry", whose window is the one that succeeded (the time to initialize),
// or the last one tried. With restart > 0, also a "restart" row: after a
// failure the pipeline starts again at t0 + restart, t0 + 2 restart, ...
// until it succeeds or the data run out; its window is then the time from t0
// to the end of the successful window, iterations the number of starts.
void evaluateRetry(RunLog& log, TraceLog* trace, const std::string& seq, double t0_label, double t0,
                   const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames, const Sophus::SE3d& T_IC,
                   const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, const ImuNoise& noise,
                   double bearing_sigma, double pixels_per_radian, const TruthFn& truth_for,
                   const FrameTruthFn& frame_truth, const FeaturelessInitConfig& lin_cfg, const Args& args) {
  const double restart = args.restart;
  DynamicInitConfig cfg;
  cfg.refine = args.refine;
  cfg.linear_covariance_scale = args.linear_covariance_scale;
  cfg.linear = lin_cfg;
  cfg.refinement.bearing_sigma = bearing_sigma;
  cfg.pixels_per_radian = pixels_per_radian;
  // Stillness disparity limit: 1 px plus 3x the pixel noise. The simulator
  // draws independent noise per frame (sigma 1 px: two still frames differ by
  // 3 px at the 90th percentile), where a real tracker's jitter is far smaller
  // and correlated; on real data args.still_noise_px is 0.
  cfg.static_init.max_disparity_px += 3.0 * args.still_noise_px;
  cfg.in_place_max_disparity_px += 3.0 * args.still_noise_px;
  auto row = [&](const DynamicInitResult& res, const std::string& method, double window, int iterations,
                 double ms) -> bool {
    if (res.reason == "data" || res.reason == "imu" || res.reason == "imu_gap") return false;  // no data for it
    const std::optional<Truth> truth = truth_for(res.keyframe_t);
    if (!truth) return false;
    RunLog::Row r;
    if (res.ok) {
      r = res.is_static     ? staticRow(res.refinement.state, *truth)
          : res.linear_only ? linearHandoffRow(res.linear, res.refinement.state, *truth)
                            : refinedRow(res.refinement, T_IC, *truth);
    }
    r.method = method;
    r.ok = res.ok;
    r.reason = res.ok ? "ok" : res.reason;
    r.init = res.ok ? (res.in_place ? "in_place" : res.is_static ? "static" : res.linear_only ? "linear" : "dynamic") : "";
    r.singular_value = res.linear.singular_value;
    r.acc_excitation = res.linear.acc_excitation;
    r.pairs = res.linear.pairs;
    r.iterations = iterations;
    r.ms = ms;
    log.add(seq, t0_label, window, res.keyframe_t.size(), r, *truth);
    return true;
  };
  auto start = std::chrono::steady_clock::now();
  const DynamicInitResult res = initializeDynamic(imu, frames, t0, T_IC, bg, ba, noise, cfg);
  if (!row(res, "retry", res.window, res.attempts, msSince(start))) return;
  if (trace) trace->add("retry", t0_label, t0, res, imu, frames, T_IC, bg, ba, noise, frame_truth);
  if (restart <= 0.0) return;
  DynamicInitResult cur = res;
  double ts = t0;
  int starts = 1;
  start = std::chrono::steady_clock::now();
  while (!cur.ok) {
    ts += restart;
    ++starts;
    cur = initializeDynamic(imu, frames, ts, T_IC, bg, ba, noise, cfg);
    if (cur.reason == "data" || cur.reason == "imu") break;
    if (trace && cur.reason != "imu_gap") {
      trace->add("restart", t0_label, ts, cur, imu, frames, T_IC, bg, ba, noise, frame_truth);
    }
  }
  if (!cur.ok) return;
  row(cur, "restart", ts - t0 + cur.window, starts, msSince(start));
}

// The frames of one window: from the first frame at or after t0 to that
// frame's time + length (so a 0.1 s window at 20 Hz holds 3 frames).
std::vector<InitFrame> windowFrames(const std::vector<InitFrame>& all, double t0, double length) {
  std::vector<InitFrame> win;
  for (const InitFrame& fr : all) {
    if (fr.t < t0) continue;
    if (!win.empty() && fr.t > win.front().t + length + 1e-3) break;  // 1 ms: timestamp jitter
    win.push_back(fr);
  }
  return win;
}


// ---------------------------------------------------------------- euroc ----

struct BiasSample {
  double t;
  Eigen::Vector3d bg, ba;
  Eigen::Vector3d v = Eigen::Vector3d::Zero();  // world velocity, when the file has it
};

// EuRoC state_groundtruth_estimate0: columns 11-13 b_w, 14-16 b_a. Empty for
// pose-only files.
std::vector<BiasSample> readGroundTruthBiases(const std::string& path) {
  std::ifstream f(path);
  std::vector<BiasSample> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::vector<double> v;
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
    if (v.size() < 17) return {};
    out.push_back({v[0] * 1e-9, Eigen::Vector3d(v[11], v[12], v[13]), Eigen::Vector3d(v[14], v[15], v[16]),
                   Eigen::Vector3d(v[8], v[9], v[10])});
  }
  return out;
}

int runEuroc(const Args& a) {
  const calib::KalibrCamera kc = calib::loadSequenceCamera(a.euroc, a.cam, a.camchain);
  if (!kc.has_T_IC) throw std::runtime_error("camera has no extrinsic (T_cam_imu / T_BS)");
  const std::vector<ImuSample> imu = io::readEurocImu((fs::path(a.euroc) / "mav0" / "imu0" / "data.csv").string());
  const eval::Trajectory gt = eval::readEurocGroundTruth(a.gt);
  // State ground truth (EuRoC layout: velocity and biases) when the file has
  // it: its velocity is the truth; pose-only files (TUM-VI) are differenced.
  const std::vector<BiasSample> state = readGroundTruthBiases(a.gt);
  const std::vector<BiasSample> biases = a.bias == "gt" ? state : std::vector<BiasSample>{};
  if (a.bias == "gt" && biases.empty()) throw std::runtime_error("--bias gt needs EuRoC's state ground truth");
  if (a.bias != "gt" && a.bias != "zero") throw std::invalid_argument("--bias " + a.bias);

  // Frames from the tracks: frame index -> (t, bearings).
  std::map<int, InitFrame> frames;
  {
    std::ifstream f(a.tracks);
    if (!f) throw std::runtime_error("cannot open " + a.tracks);
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string tok;
      std::vector<std::string> v;
      while (std::getline(ss, tok, ',')) v.push_back(tok);
      const int k = std::stoi(v[0]);
      InitFrame& fr = frames[k];
      fr.t = std::stoll(v[1]) * 1e-9;
      Eigen::Vector3d b;
      if (kc.model->unproject(Eigen::Vector2d(std::stod(v[3]), std::stod(v[4])), b)) {
        fr.bearings.emplace_back(std::stoll(v[2]), b);
      }
    }
  }
  std::vector<InitFrame> all;
  for (auto& [k, fr] : frames) all.push_back(std::move(fr));

  ImuNoise noise = a.imu_yaml.empty() ? ImuNoise{} : calib::loadKalibrImu(a.imu_yaml);
  noise.gyro_noise *= a.imu_noise_scale;
  noise.acc_noise *= a.imu_noise_scale;
  noise.gyro_bias_walk *= a.imu_noise_scale;
  noise.acc_bias_walk *= a.imu_noise_scale;
  // Bearing noise: --pixel-sigma px at the image centre, as an angle.
  double bearing_sigma = 0.0, pixels_per_radian = 0.0;
  {
    const Eigen::Vector2d c(0.5 * kc.model->width(), 0.5 * kc.model->height());
    Eigen::Vector3d b0, b1;
    if (!kc.model->unproject(c, b0) || !kc.model->unproject(c + Eigen::Vector2d(1.0, 0.0), b1)) {
      throw std::runtime_error("camera centre does not unproject");
    }
    pixels_per_radian = 1.0 / std::atan2(b0.cross(b1).norm(), b0.dot(b1));
    bearing_sigma = a.pixel_sigma / pixels_per_radian;
  }
  FeaturelessInitConfig lin_cfg;
  lin_cfg.min_acc_excitation = a.min_excitation;
  lin_cfg.triplet_scale = a.triplets;
  lin_cfg.triplet_weight = a.triplet_weight;
  lin_cfg.triplet_min_parallax_deg = a.triplet_parallax;
  fs::create_directories(a.out);
  RunLog log(fs::path(a.out) / "runs.csv");
  TraceLog trace(fs::path(a.out) / "trace.csv");
  const std::string seq = fs::path(a.euroc).filename().string() + "/" + a.cam;
  const double t_begin = std::max({all.front().t, imu.front().t, gt.front().t}) + 0.5;
  const double t_end = std::min({all.back().t, imu.back().t, gt.back().t}) - 1.5;
  int runs = 0;
  constexpr double kH = 0.01;
  // World velocity at t: the state file's (interpolated), else differenced.
  auto velocity = [&](double t, const Sophus::SE3d& Tm_, const Sophus::SE3d& Tp_) -> Eigen::Vector3d {
    if (state.empty()) return (Tp_.translation() - Tm_.translation()) / (2.0 * kH);
    const auto it = std::lower_bound(state.begin(), state.end(), t,
                                     [](const BiasSample& b, double x) { return b.t < x; });
    if (it == state.begin() || it == state.end()) return it == state.end() ? state.back().v : it->v;
    const auto prev = it - 1;
    const double u = (t - prev->t) / (it->t - prev->t);
    return prev->v + u * (it->v - prev->v);
  };
  const FrameTruthFn frame_truth = [&](double t) -> std::optional<FrameTruth> {
    Sophus::SE3d T, Tm, Tp;
    if (!eval::interpolatePose(gt, t, 0.05, T) || !eval::interpolatePose(gt, t - kH, 0.05, Tm) ||
        !eval::interpolatePose(gt, t + kH, 0.05, Tp)) {
      return std::nullopt;
    }
    return FrameTruth{((T.so3() * kc.T_IC.so3()).inverse() * kGravity).normalized(), velocity(t, Tm, Tp).norm()};
  };
  const TruthFn truth_for = [&](const std::vector<double>& times) -> std::optional<Truth> {
    Truth truth;
    Sophus::SE3d T0, Tm, Tp;
    if (times.empty() || !eval::interpolatePose(gt, times.front(), 0.05, T0) ||
        !eval::interpolatePose(gt, times.front() - kH, 0.05, Tm) ||
        !eval::interpolatePose(gt, times.front() + kH, 0.05, Tp)) {
      return std::nullopt;
    }
    truth.g_I0 = T0.so3().inverse() * kGravity;
    truth.v_I0 = T0.so3().inverse() * velocity(times.front(), Tm, Tp);
    for (double t : times) {
      Sophus::SE3d T;
      if (!eval::interpolatePose(gt, t, 0.05, T)) return std::nullopt;
      truth.p_C.push_back((T * kc.T_IC).translation());
    }
    Sophus::SE3d Tl, Tlm, Tlp;
    if (!eval::interpolatePose(gt, times.back(), 0.05, Tl) ||
        !eval::interpolatePose(gt, times.back() - kH, 0.05, Tlm) ||
        !eval::interpolatePose(gt, times.back() + kH, 0.05, Tlp)) {
      return std::nullopt;
    }
    truth.v_body_last = Tl.so3().inverse() * velocity(times.back(), Tlm, Tlp);
    return truth;
  };
  const double max_retry = DynamicInitConfig{}.max_window;
  for (double t0 = t_begin; t0 + max_retry < t_end; t0 += a.segment) {
    Eigen::Vector3d bg = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero();
    if (!biases.empty()) {
      const auto it = std::lower_bound(biases.begin(), biases.end(), t0,
                                       [](const BiasSample& b, double t) { return b.t < t; });
      if (it != biases.end()) {
        bg = it->bg;
        ba = it->ba;
      }
    }
    for (double w : a.windows) {
      const std::vector<InitFrame> win = windowFrames(all, t0, w);
      if (win.size() < 2) continue;
      evaluateWindow(log, seq, t0 - t_begin, w, imu, win, kc.T_IC, bg, ba, noise, bearing_sigma, truth_for, lin_cfg,
                     a.linear_covariance_scale);
      ++runs;
    }
    evaluateRetry(log, &trace, seq, t0 - t_begin, t0, imu, all, kc.T_IC, bg, ba, noise, bearing_sigma,
                  pixels_per_radian, truth_for, frame_truth, lin_cfg, a);
  }
  std::cout << seq << ": " << runs << " windows\n";
  return 0;
}

// ------------------------------------------------------------------ sim ----

int runSim(const Args& a) {
  const Scenario scenario = parseScenario(a.scenario);
  const CameraModelPtr camera = simulatedCamera();
  const double fx = camera->params()(0);
  FeaturelessInitConfig lin_cfg;
  lin_cfg.min_acc_excitation = a.min_excitation;
  lin_cfg.triplet_scale = a.triplets;
  lin_cfg.triplet_weight = a.triplet_weight;
  lin_cfg.triplet_min_parallax_deg = a.triplet_parallax;
  fs::create_directories(a.out);
  RunLog log(fs::path(a.out) / "runs.csv");
  int runs = 0;
  for (int seed = 1; seed <= a.seeds; ++seed) {
    SimConfig c = defaultSimConfig(scenario);
    c.duration = 6.0 * a.segment + DynamicInitConfig{}.max_window + 2.0;
    c.seed = static_cast<uint64_t>(seed);
    if (a.sim_pixel_noise >= 0.0) c.pixel_noise = a.sim_pixel_noise;
    const Simulator sim(c, makeScenario(scenario, c.duration + 1.0, static_cast<uint64_t>(seed)), camera);
    const SimData data = sim.run();
    const Sophus::SE3d T_IC(c.R_IC, c.p_IC);
    std::vector<InitFrame> all;
    for (const CameraFrame& f : data.frames) {
      InitFrame fr;
      fr.t = f.t_cam + c.time_offset;
      for (const Observation& o : f.observations) {
        Eigen::Vector3d b;
        if (camera->unproject(o.uv, b)) fr.bearings.emplace_back(o.landmark_id, b);
      }
      all.push_back(std::move(fr));
    }
    const std::string seq = a.scenario + "/seed" + std::to_string(seed);
    const TruthFn truth_for = [&](const std::vector<double>& times) -> std::optional<Truth> {
      if (times.empty()) return std::nullopt;
      Truth truth;
      const NavState s0 = sim.trueState(times.front());
      truth.g_I0 = s0.R_GI.inverse() * kGravity;
      truth.v_I0 = s0.R_GI.inverse() * s0.v_GI;
      for (double t : times) {
        const NavState st = sim.trueState(t);
        truth.p_C.push_back(st.p_GI + st.R_GI * c.p_IC);
      }
      const NavState sl = sim.trueState(times.back());
      truth.v_body_last = sl.R_GI.inverse() * sl.v_GI;
      return truth;
    };
    const double bearing_sigma = c.pixel_noise / fx;
    Args sim_args = a;
    sim_args.still_noise_px = c.pixel_noise;
    const double max_retry = DynamicInitConfig{}.max_window;
    for (double t0 = 0.5; t0 + max_retry < c.duration - 0.5; t0 += a.segment) {
      for (double w : a.windows) {
        const std::vector<InitFrame> win = windowFrames(all, t0, w);
        if (win.size() < 2) continue;
        evaluateWindow(log, seq, t0, w, data.imu, win, T_IC, c.gyro_bias0, c.acc_bias0, c.imu_noise, bearing_sigma,
                       truth_for, lin_cfg, a.linear_covariance_scale);
        ++runs;
      }
      evaluateRetry(log, nullptr, seq, t0, t0, data.imu, all, T_IC, c.gyro_bias0, c.acc_bias0, c.imu_noise,
                    bearing_sigma, fx, truth_for, {}, lin_cfg, sim_args);
    }
  }
  std::cout << a.scenario << ": " << runs << " windows\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  Args a;
  a.mode = argv[1];
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--euroc") a.euroc = next();
      else if (arg == "--cam") a.cam = next();
      else if (arg == "--camchain") a.camchain = next();
      else if (arg == "--gt") a.gt = next();
      else if (arg == "--tracks") a.tracks = next();
      else if (arg == "--imu-yaml") a.imu_yaml = next();
      else if (arg == "--bias") a.bias = next();
      else if (arg == "--scenario") a.scenario = next();
      else if (arg == "--seeds") a.seeds = std::stoi(next());
      else if (arg == "--imu-noise-scale") a.imu_noise_scale = std::stod(next());
      else if (arg == "--pixel-sigma") a.pixel_sigma = std::stod(next());
      else if (arg == "--min-excitation") a.min_excitation = std::stod(next());  // linear step's accelerometer gate
      else if (arg == "--segment") a.segment = std::stod(next());
      else if (arg == "--restart") a.restart = std::stod(next());
      else if (arg == "--triplets") a.triplets = true;
      else if (arg == "--sim-pixel-noise") a.sim_pixel_noise = std::stod(next());
      else if (arg == "--triplet-weight") a.triplet_weight = std::stod(next());
      else if (arg == "--triplet-parallax") a.triplet_parallax = std::stod(next());
      else if (arg == "--linear-cov-scale") a.linear_covariance_scale = std::stod(next());
      else if (arg == "--refine") {
        const std::string v = next();
        if (v == "always") a.refine = InitRefinementPolicy::kAlways;
        else if (v == "fallback") a.refine = InitRefinementPolicy::kFallback;
        else if (v == "never") a.refine = InitRefinementPolicy::kNever;
        else throw std::invalid_argument("--refine " + v);
      }
      else if (arg == "--windows") {
        a.windows.clear();
        std::stringstream ss(next());
        std::string tok;
        while (std::getline(ss, tok, ',')) a.windows.push_back(std::stod(tok));
        std::sort(a.windows.begin(), a.windows.end());
      }
      else if (arg == "--out") a.out = next();
      else throw std::invalid_argument("unknown argument " + arg);
    }
    if (a.out.empty()) throw std::invalid_argument("--out is required");
    if (a.mode == "euroc") return runEuroc(a);
    if (a.mode == "sim") return runSim(a);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "dynamic_init_eval: " << e.what() << '\n';
    return 1;
  }
}
