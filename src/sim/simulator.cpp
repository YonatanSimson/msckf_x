#include "msckf/sim/simulator.h"

#include "msckf/core/constants.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace msckf {

namespace {

Eigen::Vector3d gaussian3(std::mt19937_64& rng, double sigma) {
  std::normal_distribution<double> n(0.0, sigma);
  return Eigen::Vector3d(n(rng), n(rng), n(rng));
}

}  // namespace

Simulator::Simulator(const SimConfig& config, SplineTrajectory trajectory, CameraModelPtr camera)
    : config_(config), trajectory_(std::move(trajectory)), camera_(std::move(camera)) {
  // Images need poses at t_cam + t_d for t_cam in [0, duration].
  const double t_lo = std::min(0.0, config_.time_offset);
  const double t_hi = config_.duration + std::max(0.0, config_.time_offset);
  if (t_lo < trajectory_.minTime() || t_hi > trajectory_.maxTime()) {
    throw std::invalid_argument("Simulator: trajectory does not cover duration and time offset");
  }
}

ImuSample Simulator::idealImu(double t) const {
  const SplineState s = trajectory_.evaluate(t);
  ImuSample out;
  out.t = t;
  out.gyro = s.omega_I;
  out.acc = s.R_GI.inverse() * (s.a_GI - kGravity);
  return out;
}

NavState Simulator::trueState(double t) const {
  const SplineState s = trajectory_.evaluate(t);
  NavState out;
  out.R_GI = s.R_GI;
  out.p_GI = s.p_GI;
  out.v_GI = s.v_GI;
  return out;
}

SimData Simulator::run() const {
  std::mt19937_64 rng(config_.seed);
  SimData data;

  // IMU: discrete white noise sigma / sqrt(dt); bias random walk sigma * sqrt(dt)
  // per step, applied after each sample.
  const double dt = 1.0 / config_.imu_rate;
  const int n_imu = static_cast<int>(std::floor(config_.duration * config_.imu_rate + 1e-9)) + 1;
  const ImuNoise& nz = config_.imu_noise;
  const bool noisy = config_.imu_noise_enabled;
  Eigen::Vector3d bg = config_.gyro_bias0, ba = config_.acc_bias0;
  data.imu.reserve(n_imu);
  data.imu_truth.reserve(n_imu);
  for (int k = 0; k < n_imu; ++k) {
    const double t = k * dt;
    ImuSample s = idealImu(t);
    GroundTruth gt;
    gt.t = t;
    gt.state = trueState(t);
    gt.gyro_bias = bg;
    gt.acc_bias = ba;
    s.gyro += bg;
    s.acc += ba;
    if (noisy) {
      s.gyro += gaussian3(rng, nz.gyro_noise / std::sqrt(dt));
      s.acc += gaussian3(rng, nz.acc_noise / std::sqrt(dt));
      bg += gaussian3(rng, nz.gyro_bias_walk * std::sqrt(dt));
      ba += gaussian3(rng, nz.acc_bias_walk * std::sqrt(dt));
    }
    data.imu.push_back(s);
    data.imu_truth.push_back(gt);
  }

  data.landmarks = makeLandmarks(rng());

  const int n_cam = static_cast<int>(std::floor(config_.duration * config_.cam_rate + 1e-9)) + 1;
  data.frames.reserve(n_cam);
  for (int k = 0; k < n_cam; ++k) {
    data.frames.push_back(renderFrame(k / config_.cam_rate, data.landmarks, rng));
  }
  return data;
}

std::vector<Landmark> Simulator::makeLandmarks(uint64_t seed) const {
  std::mt19937_64 rng(seed);

  // Horizontal extent of the trajectory.
  Eigen::Vector2d lo(1e30, 1e30), hi(-1e30, -1e30);
  for (double t = trajectory_.minTime(); t <= trajectory_.maxTime(); t += 0.05) {
    const Eigen::Vector2d p = trajectory_.evaluate(t).p_GI.head<2>();
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  lo.array() -= config_.wall_margin;
  hi.array() += config_.wall_margin;
  const Eigen::Vector2d size = hi - lo;

  std::vector<Landmark> out;
  int64_t next_id = 0;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  auto add = [&](const Eigen::Vector3d& p) { out.push_back({next_id++, p}); };

  const int n_ground = static_cast<int>(config_.ground_density * size.x() * size.y());
  for (int i = 0; i < n_ground; ++i) {
    add(Eigen::Vector3d(lo.x() + unit(rng) * size.x(), lo.y() + unit(rng) * size.y(), 0.0));
  }

  const int n_ceiling = static_cast<int>(config_.ceiling_density * size.x() * size.y());
  for (int i = 0; i < n_ceiling; ++i) {
    add(Eigen::Vector3d(lo.x() + unit(rng) * size.x(), lo.y() + unit(rng) * size.y(),
                        config_.wall_height));
  }

  // Four walls; perimeter coordinate s runs around the rectangle.
  const double perimeter = 2.0 * (size.x() + size.y());
  const int n_wall = static_cast<int>(config_.wall_density * perimeter * config_.wall_height);
  for (int i = 0; i < n_wall; ++i) {
    double s = unit(rng) * perimeter;
    const double z = unit(rng) * config_.wall_height;
    Eigen::Vector2d xy;
    if (s < size.x()) {
      xy = Eigen::Vector2d(lo.x() + s, lo.y());
    } else if ((s -= size.x()) < size.y()) {
      xy = Eigen::Vector2d(hi.x(), lo.y() + s);
    } else if ((s -= size.y()) < size.x()) {
      xy = Eigen::Vector2d(hi.x() - s, hi.y());
    } else {
      s -= size.x();
      xy = Eigen::Vector2d(lo.x(), hi.y() - s);
    }
    add(Eigen::Vector3d(xy.x(), xy.y(), z));
  }
  return out;
}

CameraFrame Simulator::renderFrame(double t_cam, const std::vector<Landmark>& landmarks,
                                   std::mt19937_64& rng) const {
  CameraFrame frame;
  frame.t_cam = t_cam;
  const NavState x = trueState(t_cam + config_.time_offset);
  // p_C = R_IC^T (R_GI^T (p_G - p_GI) - p_IC)
  const Sophus::SO3d R_CG = config_.R_IC.inverse() * x.R_GI.inverse();
  const Eigen::Vector3d p_CG = -(config_.R_IC.inverse() * (x.R_GI.inverse() * x.p_GI + config_.p_IC));
  std::normal_distribution<double> pixel(0.0, config_.pixel_noise);

  for (const Landmark& lm : landmarks) {
    const Eigen::Vector3d p_C = R_CG * lm.p_G + p_CG;
    if (p_C.norm() > config_.max_depth) continue;
    Eigen::Vector2d uv;
    if (!camera_->project(p_C, uv) || !camera_->inImage(uv)) continue;
    Observation obs;
    obs.landmark_id = lm.id;
    obs.uv_true = uv;
    obs.uv = uv + Eigen::Vector2d(pixel(rng), pixel(rng));
    if (!camera_->validPixel(obs.uv)) continue;
    frame.observations.push_back(obs);
  }
  return frame;
}

}  // namespace msckf
