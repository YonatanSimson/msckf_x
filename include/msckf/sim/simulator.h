#pragma once

#include "msckf/camera/camera_model.h"
#include "msckf/imu/imu_types.h"
#include "msckf/sim/spline.h"

#include <cstdint>
#include <random>
#include <vector>

namespace msckf {

// Camera looking along IMU +x (forward), image x to the right (IMU -y), image y
// down (IMU -z), pitched down by the given angle.
inline Sophus::SO3d forwardCameraRotation(double pitch_down_rad) {
  Eigen::Matrix3d R;  // columns: camera x, y, z axes expressed in the IMU frame
  R << 0, 0, 1,
      -1, 0, 0,
       0, -1, 0;
  return Sophus::SO3d::rotY(pitch_down_rad) * Sophus::SO3d(R);
}

struct SimConfig {
  double duration = 60.0;   // s, simulated from t = 0
  double imu_rate = 400.0;  // Hz
  double cam_rate = 20.0;   // Hz

  ImuNoise imu_noise;
  bool imu_noise_enabled = true;
  Eigen::Vector3d gyro_bias0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias0 = Eigen::Vector3d::Zero();

  // Camera pose in the IMU frame.
  Sophus::SO3d R_IC = forwardCameraRotation(15.0 * M_PI / 180.0);
  Eigen::Vector3d p_IC = Eigen::Vector3d(0.05, -0.03, 0.02);
  double time_offset = 0.0;  // t_d, with t_imu = t_cam + t_d
  double pixel_noise = 1.0;  // px, per axis
  double max_depth = 40.0;   // m, observations beyond this are dropped

  // Landmarks: ground plane z = 0, four walls around the trajectory and an
  // optional ceiling on top of them (density 0 disables it).
  double ground_density = 4.0;   // points per m^2
  double wall_density = 4.0;     // points per m^2
  double ceiling_density = 4.0;  // points per m^2
  double wall_margin = 4.0;      // m, wall distance beyond the trajectory's extent
  double wall_height = 4.0;      // m

  uint64_t seed = 1;
};

struct Landmark {
  int64_t id = 0;
  Eigen::Vector3d p_G = Eigen::Vector3d::Zero();
};

struct Observation {
  int64_t landmark_id = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();       // with pixel noise
  Eigen::Vector2d uv_true = Eigen::Vector2d::Zero();  // exact projection
};

struct CameraFrame {
  double t_cam = 0.0;  // camera timestamp; the image was taken at t_cam + t_d
  std::vector<Observation> observations;
};

// Ground truth at an IMU sample time.
struct GroundTruth {
  double t = 0.0;
  NavState state;
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
};

struct SimData {
  std::vector<ImuSample> imu;
  std::vector<GroundTruth> imu_truth;  // one per IMU sample
  std::vector<CameraFrame> frames;
  std::vector<Landmark> landmarks;
};

class Simulator {
 public:
  Simulator(const SimConfig& config, SplineTrajectory trajectory, CameraModelPtr camera);

  const SimConfig& config() const { return config_; }
  const SplineTrajectory& trajectory() const { return trajectory_; }
  const CameraModel& camera() const { return *camera_; }

  // Exact, noise- and bias-free IMU reading at any time in the trajectory range.
  ImuSample idealImu(double t) const;

  // Exact navigation state at any time in the trajectory range.
  NavState trueState(double t) const;

  // Runs the full simulation: IMU with noise and bias walk, landmarks, frames.
  SimData run() const;

 private:
  std::vector<Landmark> makeLandmarks(uint64_t seed) const;
  CameraFrame renderFrame(double t_cam, const std::vector<Landmark>& landmarks,
                          std::mt19937_64& rng) const;

  SimConfig config_;
  SplineTrajectory trajectory_;
  CameraModelPtr camera_;
};

}  // namespace msckf
