#pragma once

#include "msckf/camera/camera_model.h"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace msckf::calib {

// One camera of a Kalibr camchain.yaml (camchain.yaml, camchain-imucam.yaml).
struct KalibrCamera {
  std::string name;        // cam0, cam1, ...
  std::string topic;       // rostopic
  std::string model_name;  // Kalibr's "<camera_model>-<distortion_model>", e.g. pinhole-equi
  CameraModelPtr model;
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

}  // namespace msckf::calib
