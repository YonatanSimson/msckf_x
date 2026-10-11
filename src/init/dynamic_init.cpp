#include "msckf/init/dynamic_init.h"

#include "msckf/core/constants.h"
#include "msckf/imu/preintegration.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace msckf {

namespace {

// Median angle (rad) between the bearings of the features seen in both the
// first and the last frame; < 0 when they share none.
double medianDisparity(const std::vector<InitFrame>& frames) {
  if (frames.size() < 2) return -1.0;
  std::vector<double> d;
  for (const auto& [id, b0] : frames.front().bearings) {
    for (const auto& [id1, b1] : frames.back().bearings) {
      if (id1 == id) {
        d.push_back(std::atan2(b0.cross(b1).norm(), b0.dot(b1)));
        break;
      }
    }
  }
  if (d.empty()) return -1.0;
  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
  return d[d.size() / 2];
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
  for (const auto& [id, b0] : frames.front().bearings) {
    for (const auto& [id1, b1] : frames.back().bearings) {
      if (id1 == id) {
        const Eigen::Vector3d p = R_CeCs * b0;
        d.push_back(std::atan2(p.cross(b1).norm(), p.dot(b1)) * config.pixels_per_radian);
        break;
      }
    }
  }
  if (d.size() < 10) return std::nullopt;  // too few common features to judge
  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
  const double disparity = d[d.size() / 2];
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
    if (imu.empty() || imu.back().t < win.back().t) {
      res.reason = "imu";
      break;
    }
    {
      // IMU continuity over the window.
      double prev = -1.0, gap = 0.0;
      for (const ImuSample& smp : imu) {
        if (smp.t < win.front().t - config.max_imu_gap) continue;
        if (prev >= 0.0) gap = std::max(gap, smp.t - prev);
        prev = smp.t;
        if (smp.t > win.back().t) break;
      }
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
        const double d = medianDisparity(tail);
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
    res.refinement = refineInit(imu, kf, T_IC, res.linear, gyro_bias, acc_bias, noise, config.refinement);
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
