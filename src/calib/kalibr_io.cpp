#include "msckf/calib/kalibr_io.h"

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/eucm.h"
#include "msckf/camera/kannala_brandt.h"
#include "msckf/camera/omni_radtan.h"
#include "msckf/camera/pinhole_radtan.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace msckf::calib {

namespace {

void requireSize(const std::vector<double>& v, size_t n, const std::string& what) {
  if (v.size() != n) {
    throw std::invalid_argument(what + ": expected " + std::to_string(n) + " values, got " +
                                std::to_string(v.size()));
  }
}

}  // namespace

CameraModelPtr makeKalibrCamera(const std::string& camera_model, const std::string& distortion_model,
                                const std::vector<double>& in, const std::vector<double>& d,
                                int width, int height) {
  // camchain files spell out "equidistant"; the command-line names use "equi".
  const std::string distortion = distortion_model == "equidistant" ? "equi" : distortion_model;
  const std::string name = camera_model + "-" + distortion;
  if (name == "pinhole-radtan") {
    requireSize(in, 4, name + " intrinsics");
    requireSize(d, 4, name + " distortion");
    Eigen::VectorXd p(9);
    p << in[0], in[1], in[2], in[3], d[0], d[1], d[2], d[3], 0.0;
    return std::make_shared<PinholeRadtan>(width, height, p);
  }
  if (name == "pinhole-equi") {
    requireSize(in, 4, name + " intrinsics");
    requireSize(d, 4, name + " distortion");
    Eigen::VectorXd p(8);
    p << in[0], in[1], in[2], in[3], d[0], d[1], d[2], d[3];
    return std::make_shared<KannalaBrandt>(width, height, p);
  }
  if (name == "ds-none") {
    requireSize(in, 6, name + " intrinsics");
    Eigen::VectorXd p(6);
    p << in[2], in[3], in[4], in[5], in[0], in[1];
    return std::make_shared<DoubleSphere>(width, height, p);
  }
  if (name == "eucm-none") {
    requireSize(in, 6, name + " intrinsics");
    Eigen::VectorXd p(6);
    p << in[2], in[3], in[4], in[5], in[0], in[1];
    return std::make_shared<Eucm>(width, height, p);
  }
  if (name == "omni-radtan") {
    requireSize(in, 5, name + " intrinsics");
    requireSize(d, 4, name + " distortion");
    Eigen::VectorXd p(9);
    p << in[1], in[2], in[3], in[4], in[0], d[0], d[1], d[2], d[3];
    return std::make_shared<OmniRadtan>(width, height, p);
  }
  throw std::invalid_argument("unsupported Kalibr camera model " + name);
}

std::vector<KalibrCamera> loadKalibrCamchain(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  std::vector<KalibrCamera> cams;
  for (int i = 0;; ++i) {
    const std::string key = "cam" + std::to_string(i);
    const YAML::Node n = root[key];
    if (!n) break;
    const auto res = n["resolution"].as<std::vector<int>>();
    const std::string camera_model = n["camera_model"].as<std::string>();
    const std::string distortion_model =
        n["distortion_model"] ? n["distortion_model"].as<std::string>() : "none";
    const auto distortion =
        n["distortion_coeffs"] ? n["distortion_coeffs"].as<std::vector<double>>() : std::vector<double>{};
    KalibrCamera cam;
    cam.name = key;
    cam.topic = n["rostopic"] ? n["rostopic"].as<std::string>() : "";
    cam.model = makeKalibrCamera(camera_model, distortion_model,
                                 n["intrinsics"].as<std::vector<double>>(), distortion, res.at(0),
                                 res.at(1));
    cam.model_name = camera_model + "-" + (distortion_model == "equidistant" ? "equi" : distortion_model);
    cams.push_back(cam);
  }
  if (cams.empty()) throw std::runtime_error(path + ": no cam0 entry");
  return cams;
}

ImuNoise loadKalibrImu(const std::string& path) {
  const YAML::Node n = YAML::LoadFile(path);
  ImuNoise noise;
  noise.acc_noise = n["accelerometer_noise_density"].as<double>();
  noise.acc_bias_walk = n["accelerometer_random_walk"].as<double>();
  noise.gyro_noise = n["gyroscope_noise_density"].as<double>();
  noise.gyro_bias_walk = n["gyroscope_random_walk"].as<double>();
  return noise;
}

}  // namespace msckf::calib
