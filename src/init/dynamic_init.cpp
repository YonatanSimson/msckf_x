#include "msckf/init/dynamic_init.h"

#include "msckf/core/constants.h"
#include "msckf/imu/preintegration.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace msckf {

namespace {

// The `quantile` of x (0..1, nearest rank); x is reordered.
double quantile(std::vector<double>& x, double q) {
  const size_t i = std::min(x.size() - 1, static_cast<size_t>(q * static_cast<double>(x.size())));
  std::nth_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(i), x.end());
  return x[i];
}

// Bearings of the features seen in both frames, (in a, in b).
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> commonBearings(const InitFrame& a, const InitFrame& b) {
  std::map<int64_t, Eigen::Vector3d> in_b(b.bearings.begin(), b.bearings.end());
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> out;
  for (const auto& [id, ba] : a.bearings) {
    if (const auto it = in_b.find(id); it != in_b.end()) out.emplace_back(ba, it->second);
  }
  return out;
}

// Angle (rad) between the bearings of the features seen in both the first and
// the last frame, at the given quantile; < 0 when they share none.
double disparityQuantile(const std::vector<InitFrame>& frames, double q) {
  if (frames.size() < 2) return -1.0;
  std::vector<double> d;
  for (const auto& [b0, b1] : commonBearings(frames.front(), frames.back())) {
    d.push_back(std::atan2(b0.cross(b1).norm(), b0.dot(b1)));
  }
  if (d.empty()) return -1.0;
  return quantile(d, q);
}

}  // namespace

std::optional<ImuInitState> inPlaceInitialize(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                              const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                              const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                              const DynamicInitConfig& config, double* disparity_px) {
  if (disparity_px) *disparity_px = -1.0;
  if (frames.size() < 2 || config.pixels_per_radian <= 0.0) return std::nullopt;
  const double t_s = frames.front().t, t_e = frames.back().t;
  ImuPreintegration pre(gyro_bias, acc_bias, noise);
  pre.integrateRange(imu, t_s, t_e);
  const double T = pre.deltaTime();

  // Rotation-compensated disparity: the first frame's bearing rotated into
  // the last camera frame, against the last frame's bearing.
  const Eigen::Matrix3d R_CeCs = (T_IC.so3().inverse() * pre.deltaR().inverse() * T_IC.so3()).matrix();
  std::vector<double> d;
  for (const auto& [b0, b1] : commonBearings(frames.front(), frames.back())) {
    const Eigen::Vector3d p = R_CeCs * b0;
    d.push_back(std::atan2(p.cross(b1).norm(), p.dot(b1)) * config.pixels_per_radian);
  }
  if (d.size() < 10) return std::nullopt;  // too few common features to judge
  const double disparity = quantile(d, config.still_disparity_quantile);
  if (disparity_px) *disparity_px = disparity;
  if (disparity > config.in_place_max_disparity_px) return std::nullopt;

  // Mean specific force in the last IMU frame: dR^T dv / T = R_e^T (v_e - v_s
  // - g T) / T, i.e. "up" times |g| plus the velocity change over T.
  const Eigen::Vector3d f = pre.deltaR().inverse() * pre.deltaV() / T;
  const double f_norm = f.norm();
  if (std::abs(f_norm - kGravity.norm()) > config.static_init.max_gravity_error) return std::nullopt;
  const Eigen::Vector3d z = f / f_norm;

  ImuInitState s;
  s.t = t_e;
  s.nav.R_GI = attitudeFromUp(z);
  s.gyro_bias = gyro_bias;
  s.acc_bias = acc_bias;

  // Tilt error as in staticInitialize, from the averaged accelerometer noise,
  // the bias prior and the unknown velocity change (v_s, v_e each
  // in_place_velocity_sigma: (v_e - v_s) / T).
  using namespace imu_err;
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = Sophus::SO3d::hat(z);
  const double sv = config.in_place_velocity_sigma;
  const Eigen::Matrix3d cov_f = (noise.acc_noise * noise.acc_noise / T + 2.0 * sv * sv / (T * T)) * I3;
  const double sba = config.static_init.acc_bias_sigma, sbg = config.refinement.gyro_bias_sigma;
  const Eigen::Matrix3d cov_ba = sba * sba * I3;
  const double yaw = config.static_init.yaw_sigma;
  s.P.setZero();
  s.P.block<3, 3>(kTheta, kTheta) = S * (cov_f + cov_ba) * S.transpose() / (f_norm * f_norm) + yaw * yaw * z * z.transpose();
  s.P.block<3, 3>(kTheta, kBa) = S * cov_ba / f_norm;
  s.P.block<3, 3>(kBa, kTheta) = s.P.block<3, 3>(kTheta, kBa).transpose();
  s.P.block<3, 3>(kBa, kBa) = cov_ba;
  s.P.block<3, 3>(kBg, kBg) = sbg * sbg * I3;
  s.P.block<3, 3>(kP, kP) = config.static_init.position_sigma * config.static_init.position_sigma * I3;
  s.P.block<3, 3>(kV, kV) = sv * sv * I3;
  return s;
}

std::optional<ImuInitState> linearHandoff(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& keyframes,
                                          const Sophus::SE3d& T_IC, const FeaturelessInitResult& linear,
                                          const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias,
                                          const ImuNoise& noise, const DynamicInitConfig& config) {
  using namespace imu_err;
  if (!linear.has_covariance || keyframes.size() < 2) return std::nullopt;
  const double gn = linear.g_I0.norm();
  ImuPreintegration pre(gyro_bias, acc_bias, noise);
  pre.integrateRange(imu, keyframes.front().t, keyframes.back().t);
  const double T = pre.deltaTime();
  NavState s0;
  s0.R_GI = attitudeFromUp(-linear.g_I0 / gn);
  s0.v_GI = s0.R_GI * linear.v_I0;
  ImuInitState st;
  st.t = keyframes.back().t;
  st.nav = pre.predict(s0, gyro_bias, acc_bias);
  st.gyro_bias = gyro_bias;
  st.acc_bias = acc_bias;

  // Sensitivity of the linear solution y = (v_I0, g_I0) to the bias (it used
  // the deltas at the prior bias), by central differences.
  Eigen::Matrix<double, 6, 6> S_b;
  for (int i = 0; i < 6; ++i) {
    const double eps = 1e-4;
    Eigen::Matrix<double, 6, 1> y[2];
    for (int k = 0; k < 2; ++k) {
      Eigen::Matrix<double, 6, 1> db = Eigen::Matrix<double, 6, 1>::Zero();
      db(i) = k == 0 ? eps : -eps;
      const FeaturelessInitResult r = featurelessInit(imu, keyframes, T_IC, gyro_bias + db.head<3>(),
                                                      acc_bias + db.tail<3>(), noise, config.linear);
      if (!r.ok) return std::nullopt;
      y[k] << r.v_I0, r.g_I0;
    }
    S_b.col(i) = (y[0] - y[1]) / (2.0 * eps);
  }

  // First-order map from y to the hand-off errors (dtheta body, dp global,
  // dv global). The first frame's tilt error is dtheta0 = -[g_I0]x dg / |g|^2
  // (R_GI0 g_I0 = g for any estimate, so gravity itself cancels):
  //   dtheta = R_l^T dtheta0                                (R_l: I0 -> last)
  //   dp_G = R_GI0 (dv_I0 T - [v_I0 T + dp_pre]x dtheta0)
  //   dv_G = R_GI0 (dv_I0 - [v_I0 + dv_pre]x dtheta0)
  const Eigen::Matrix3d Rl = pre.deltaR().matrix(), R_GI0 = s0.R_GI.matrix();
  const Eigen::Matrix3d dth0_dg = -Sophus::SO3d::hat(linear.g_I0) / (gn * gn);
  Eigen::Matrix<double, 9, 6> L = Eigen::Matrix<double, 9, 6>::Zero();
  L.block<3, 3>(kTheta, 3) = Rl.transpose() * dth0_dg;
  L.block<3, 3>(kP, 0) = R_GI0 * T;
  L.block<3, 3>(kP, 3) = -R_GI0 * Sophus::SO3d::hat(linear.v_I0 * T + pre.deltaP()) * dth0_dg;
  L.block<3, 3>(kV, 0) = R_GI0;
  L.block<3, 3>(kV, 3) = -R_GI0 * Sophus::SO3d::hat(linear.v_I0 + pre.deltaV()) * dth0_dg;
  // The linear covariance is over (v_I0, du), g_I0 = |g| normalize(u + B du).
  Eigen::Matrix<double, 6, 5> M = Eigen::Matrix<double, 6, 5>::Zero();
  M.topLeftCorner<3, 3>().setIdentity();
  M.bottomRightCorner<3, 2>() = gn * linear.gravity_basis;
  // The bias error moves the window's deltas (with y fixed) and y itself.
  Eigen::Matrix<double, 9, 6> A = L * S_b;
  A.block<3, 3>(kTheta, 0) += pre.dR_dbg();
  A.block<3, 3>(kP, 0) += R_GI0 * pre.dP_dbg();
  A.block<3, 3>(kP, 3) += R_GI0 * pre.dP_dba();
  A.block<3, 3>(kV, 0) += R_GI0 * pre.dV_dbg();
  A.block<3, 3>(kV, 3) += R_GI0 * pre.dV_dba();
  const double sbg = config.refinement.gyro_bias_sigma, sba = config.refinement.acc_bias_sigma;
  Eigen::Matrix<double, 6, 6> Pb = Eigen::Matrix<double, 6, 6>::Zero();
  Pb.diagonal() << sbg * sbg, sbg * sbg, sbg * sbg, sba * sba, sba * sba, sba * sba;
  // Preintegration noise: dtheta is already in the last frame (right
  // perturbation); dp and dv are in I0.
  Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Identity();
  G.block<3, 3>(kP, kP) = R_GI0;
  G.block<3, 3>(kV, kV) = R_GI0;

  static_assert(kTheta == 0 && kP == 3 && kV == 6 && kBg == 9 && kBa == 12);
  st.P.setZero();
  st.P.topLeftCorner<9, 9>() = config.linear_covariance_scale * L * M * linear.covariance * M.transpose() * L.transpose() +
                               A * Pb * A.transpose() + G * pre.covariance() * G.transpose();
  st.P.block<9, 6>(0, kBg) = A * Pb;
  st.P.block<6, 9>(kBg, 0) = (A * Pb).transpose();
  st.P.block<6, 6>(kBg, kBg) = Pb;
  // Gauge: yaw (about global z, in the body frame) and the first frame's position.
  const Eigen::Vector3d z_b = st.nav.R_GI.inverse() * Eigen::Vector3d::UnitZ();
  const double ys = config.refinement.gauge_sigma;
  st.P.block<3, 3>(kTheta, kTheta) += ys * ys * z_b * z_b.transpose();
  st.P.block<3, 3>(kP, kP) += ys * ys * Eigen::Matrix3d::Identity();
  return st;
}

bool passesScaleGate(const ImuInitState& state, const DynamicInitConfig& config, double* velocity_sigma) {
  const Eigen::Matrix3d Pv = state.P.block<3, 3>(imu_err::kV, imu_err::kV);
  const double sigma = std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Pv).eigenvalues().maxCoeff());
  if (velocity_sigma) *velocity_sigma = sigma;
  return sigma <= std::max(config.max_relative_velocity_sigma * state.nav.v_GI.norm(), config.min_velocity_sigma);
}

DynamicInitResult initializeDynamic(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                    double t_start, const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                    const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                    const DynamicInitConfig& config) {
  if (!(config.growth > 0.0)) throw std::invalid_argument("initializeDynamic: growth must be > 0");
  DynamicInitResult res;
  for (double w = config.min_window; w <= config.max_window + 1e-9; w += config.growth) {
    // The window: from the first frame at or after t_start to its time + w
    // (1 ms tolerance for timestamp jitter).
    std::vector<InitFrame> win;
    bool complete = false;  // a frame exists past the window: it is fully observed
    for (const InitFrame& f : frames) {
      if (f.t < t_start) continue;
      if (!win.empty() && f.t > win.front().t + w + 1e-3) {
        complete = true;
        break;
      }
      win.push_back(f);
    }
    if (win.size() < 2 || !complete) {
      res.reason = "data";  // the frames run out before this window ends
      break;
    }
    if (imu.empty() || imu.front().t > win.front().t || imu.back().t < win.back().t) {
      res.reason = "imu";
      break;
    }
    {
      // IMU continuity over the window, from the last sample at or before its
      // start to the first after its end.
      const auto later = [](double v, const ImuSample& x) { return v < x.t; };
      auto it = std::upper_bound(imu.begin(), imu.end(), win.front().t, later) - 1;
      double gap = 0.0;
      for (; it + 1 != imu.end() && it->t <= win.back().t; ++it) gap = std::max(gap, (it + 1)->t - it->t);
      if (gap > config.max_imu_gap) {
        res.reason = "imu_gap";
        res.window = w;
        break;  // a longer window would still contain the gap
      }
    }
    ++res.attempts;
    res.window = w;
    if (config.try_static) {
      StaticInitConfig sc = config.static_init;
      sc.window = std::min(w, sc.window);
      sc.noise = noise;
      sc.acc_bias_prior = acc_bias;
      std::vector<InitFrame> tail;
      for (const InitFrame& f : win) {
        if (f.t >= win.back().t - sc.window - 1e-3) tail.push_back(f);
      }
      std::optional<double> disparity;
      if (config.pixels_per_radian > 0.0) {
        const double d = disparityQuantile(tail, config.still_disparity_quantile);
        // No common feature over the window: no evidence of stillness.
        disparity = d < 0.0 ? 1e9 : d * config.pixels_per_radian;
      }
      std::optional<ImuInitState> st = staticInitialize(imu, win.back().t, sc, disparity);
      if (!st && config.try_in_place) {
        st = inPlaceInitialize(imu, tail, T_IC, gyro_bias, acc_bias, noise, config);
        res.in_place = st.has_value();
      }
      if (st) {
        res.ok = true;
        res.is_static = true;
        res.linear_only = false;  // earlier attempts' fields describe failures
        res.velocity_sigma = 0.0;
        res.linear = FeaturelessInitResult{};
        res.reason.clear();
        res.keyframe_t = {win.back().t};
        res.refinement = InitRefinementResult{};
        res.refinement.ok = true;
        res.refinement.state = *st;
        return res;
      }
    }
    std::vector<InitFrame> kf;
    for (int k : selectKeyframes(imu, win, T_IC, gyro_bias)) kf.push_back(win[k]);
    res.keyframe_t.clear();
    for (const InitFrame& f : kf) res.keyframe_t.push_back(f.t);
    res.linear = featurelessInit(imu, kf, T_IC, gyro_bias, acc_bias, noise, config.linear);
    if (!res.linear.ok) {
      res.reason = res.linear.reason;
      continue;
    }
    res.linear_only = false;
    if (config.refine != InitRefinementPolicy::kNever) {
      res.refinement = refineInit(imu, kf, T_IC, res.linear, gyro_bias, acc_bias, noise, config.refinement);
    }
    if (config.refine == InitRefinementPolicy::kNever ||
        (!res.refinement.ok && config.refine == InitRefinementPolicy::kFallback)) {
      const std::string why = res.refinement.ok ? "" : res.refinement.reason;
      res.refinement = InitRefinementResult{};
      if (auto st = linearHandoff(imu, kf, T_IC, res.linear, gyro_bias, acc_bias, noise, config)) {
        res.refinement.ok = true;
        res.refinement.state = *st;
        res.linear_only = true;
      } else {
        res.refinement.reason = why.empty() ? "covariance" : why;
      }
    }
    if (!res.refinement.ok) {
      res.reason = res.refinement.reason;
      continue;
    }
    if (!passesScaleGate(res.refinement.state, config, &res.velocity_sigma)) {
      res.reason = "scale";
      continue;
    }
    res.ok = true;
    res.reason.clear();
    return res;
  }
  return res;
}

}  // namespace msckf
