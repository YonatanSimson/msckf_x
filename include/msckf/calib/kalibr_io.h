#pragma once

#include "msckf/camera/camera_model.h"
#include "msckf/imu/imu_types.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <string>
#include <vector>

namespace msckf::calib {

// One camera of a Kalibr camchain.yaml (camchain.yaml, camchain-imucam.yaml).
struct KalibrCamera {
  std::string name;        // cam0, cam1, ...
  std::string topic;       // rostopic
  std::string model_name;  // Kalibr's "<camera_model>-<distortion_model>", e.g. pinhole-equi
  CameraModelPtr model;
  Sophus::SE3d T_IC;       // camera pose in the IMU frame (inverse of Kalibr's T_cam_imu)
  bool has_T_IC = false;   // camchain.yaml has no T_cam_imu
};

// Builds our camera model from Kalibr's description:
//   pinhole-radtan  [fu fv pu pv] [k1 k2 r1 r2]  -> PinholeRadtan (k3 = 0)
//   pinhole-equi    [fu fv pu pv] [k1 k2 k3 k4]  -> KannalaBrandt
//   ds-none         [xi alpha fu fv pu pv]       -> DoubleSphere
//   eucm-none       [alpha beta fu fv pu pv]     -> Eucm
//   omni-radtan     [xi fu fv pu pv] [k1 k2 r1 r2] -> OmniRadtan
// Throws std::invalid_argument for anything else.
CameraModelPtr makeKalibrCamera(const std::string& camera_model, const std::string& distortion_model,
                                const std::vector<double>& intrinsics,
                                const std::vector<double>& distortion, int width, int height);

std::vector<KalibrCamera> loadKalibrCamchain(const std::string& path);

// A camera from a EuRoC-format mav0/camN/sensor.yaml (camera_model pinhole,
// distortion_model radial-tangential or equidistant, T_BS = camera pose in
// the body/IMU frame). The name is the directory name.
KalibrCamera loadEurocCamera(const std::string& sensor_yaml);

// Noise densities from a Kalibr imu.yaml (accelerometer_noise_density, ...).
ImuNoise loadKalibrImu(const std::string& path);

}  // namespace msckf::calib
