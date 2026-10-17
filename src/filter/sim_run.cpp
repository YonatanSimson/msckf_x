#include "msckf/filter/sim_run.h"

#include <algorithm>
#include <Eigen/Cholesky>

#include <random>

namespace msckf::filter {

namespace {

const GroundTruth& truthAt(const SimData& d, double t) {
  const auto it = std::lower_bound(d.imu_truth.begin(), d.imu_truth.end(), t - 1e-9,
                                   [](const GroundTruth& g, double v) { return g.t < v; });
  return it == d.imu_truth.end() ? d.imu_truth.back() : *it;
}

}  // namespace

std::vector<SimEpoch> runFilterOnSimulation(const Simulator& sim, const CameraModelPtr& camera, const SimData& data,
                                            const MsckfOptions& options, const SimRunConfig& config) {
  using namespace imu_err;
  const SimConfig& sc = sim.config();
  std::mt19937_64 rng(config.seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  const double k = config.perturb ? 1.0 : 0.0;
  const auto draw = [&](double sigma) -> Eigen::Vector3d { return Eigen::Vector3d(n01(rng), n01(rng), n01(rng)) * sigma * k; };

  size_t f0 = 0;
  while (f0 < data.frames.size() && data.frames[f0].t_cam < config.t_start) ++f0;
  if (f0 == data.frames.size()) return {};
  const double t0 = data.frames[f0].t_cam + sc.time_offset;
  const GroundTruth& g0 = truthAt(data, t0);

  // true = estimate (+) error, so estimate = truth (-) error.
  ImuInitState init;
  init.t = t0;
  init.nav = sim.trueState(t0);
  init.nav.R_GI = init.nav.R_GI * Sophus::SO3d::exp(-draw(config.sigma_theta));
  init.nav.p_GI -= draw(config.sigma_p);
  init.nav.v_GI -= draw(config.sigma_v);
  init.gyro_bias = g0.gyro_bias - draw(config.sigma_bg);
  init.acc_bias = g0.acc_bias - draw(config.sigma_ba);
  Eigen::Matrix<double, kDim, 1> sd;
  sd << Eigen::Vector3d::Constant(config.sigma_theta), Eigen::Vector3d::Constant(config.sigma_p),
      Eigen::Vector3d::Constant(config.sigma_v), Eigen::Vector3d::Constant(config.sigma_bg),
      Eigen::Vector3d::Constant(config.sigma_ba);
  init.P = sd.cwiseAbs2().asDiagonal();

  const bool pe = options.state.estimate_extrinsic, pt = options.state.estimate_time_offset;
  const Eigen::Vector3d ext_theta = pe ? draw(config.sigma_ext_theta) : Eigen::Vector3d::Zero();
  const Eigen::Vector3d ext_p = pe ? draw(config.sigma_ext_p) : Eigen::Vector3d::Zero();
  const Sophus::SO3d R_IC = sc.R_IC * Sophus::SO3d::exp(-ext_theta);
  const Eigen::Vector3d p_IC = sc.p_IC - ext_p;
  const double t_d = sc.time_offset - (pt ? config.sigma_td * n01(rng) * k : 0.0);
  Eigen::Matrix<double, 7, 1> cd;
  cd << Eigen::Vector3d::Constant(config.sigma_ext_theta), Eigen::Vector3d::Constant(config.sigma_ext_p),
      config.sigma_td;
  const Eigen::Matrix<double, 7, 7> calib_cov = cd.cwiseAbs2().asDiagonal();

  Msckf filter(camera, init, R_IC, p_IC, t_d, calib_cov, options);
  size_t imu_i = 0;
  while (imu_i < data.imu.size() && data.imu[imu_i].t <= t0) ++imu_i;
  if (imu_i > 0) --imu_i;

  std::vector<SimEpoch> out;
  for (size_t f = f0 + 1; f < data.frames.size(); ++f) {
    const CameraFrame& frame = data.frames[f];
    const double t_img = frame.t_cam + filter.state().t_d;
    while (imu_i < data.imu.size() && (imu_i == 0 || data.imu[imu_i - 1].t <= t_img + 0.01)) {
      filter.addImu(data.imu[imu_i++]);
    }
    if (imu_i == data.imu.size() && data.imu.back().t < t_img) break;
    std::vector<FeatureObservation> obs;
    obs.reserve(frame.observations.size());
    for (const Observation& o : frame.observations) obs.push_back({o.landmark_id, o.uv});
    const FrameResult r = filter.addImage(frame.t_cam, obs);

    const FilterState& s = filter.state();
    const Clone& c = s.clones().back();
    const double t_true = frame.t_cam + sc.time_offset;
    const NavState truth = sim.trueState(t_true);
    SimEpoch e;
    e.t = frame.t_cam;
    e.T_est = Sophus::SE3d(c.R_GI, c.p_GI);
    e.T_true = Sophus::SE3d(truth.R_GI, truth.p_GI);
    e.cov_pose = s.P.block<6, 6>(s.cloneIndex(s.clones().size() - 1), s.cloneIndex(s.clones().size() - 1));
    e.v_est = s.nav.v_GI;
    e.v_true = truth.v_GI;
    e.cov_v = s.P.block<3, 3>(kV, kV);
    const GroundTruth& gb = truthAt(data, s.t);
    const Eigen::Vector3d dbg = gb.gyro_bias - s.gyro_bias, dba = gb.acc_bias - s.acc_bias;
    e.nees_bg = dbg.dot(s.P.block<3, 3>(kBg, kBg).ldlt().solve(dbg));
    e.nees_ba = dba.dot(s.P.block<3, 3>(kBa, kBa).ldlt().solve(dba));
    e.ext_theta_err = (s.R_IC.inverse() * sc.R_IC).log();
    e.ext_p_err = sc.p_IC - s.p_IC;
    e.td_err = sc.time_offset - s.t_d;
    e.t_state = s.t;
    e.nav_prior = s.nav_fej;
    e.nav_post = s.nav;
    e.gyro_bias = s.gyro_bias;
    e.acc_bias = s.acc_bias;
    e.observations = static_cast<int>(obs.size());
    e.update = r.update;
    e.time_ms = r.time_ms;
    out.push_back(e);
  }
  return out;
}

}  // namespace msckf::filter
