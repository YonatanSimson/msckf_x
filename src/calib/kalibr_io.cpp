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
    // k1 k2 p1 p2, optionally k3 (KITTI's calibration has it).
    if (d.size() != 5) requireSize(d, 4, name + " distortion");
    Eigen::VectorXd p(9);
    p << in[0], in[1], in[2], in[3], d[0], d[1], d[2], d[3], d.size() == 5 ? d[4] : 0.0;
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
    if (n["T_cam_imu"]) {
      Eigen::Matrix4d T;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) T(r, c) = n["T_cam_imu"][r][c].as<double>();
      }
      const Sophus::SE3d T_CI(Sophus::makeRotationMatrix(T.topLeftCorner<3, 3>()), T.topRightCorner<3, 1>());
      cam.T_IC = T_CI.inverse();
      cam.has_T_IC = true;
    }
    cams.push_back(cam);
  }
  if (cams.empty()) throw std::runtime_error(path + ": no cam0 entry");
  return cams;
}

KalibrCamera loadEurocCamera(const std::string& sensor_yaml) {
  const YAML::Node n = YAML::LoadFile(sensor_yaml);
  const auto res = n["resolution"].as<std::vector<int>>();
  std::string distortion = n["distortion_model"].as<std::string>();
  if (distortion == "radial-tangential") distortion = "radtan";
  KalibrCamera cam;
  const std::string dir = sensor_yaml.substr(0, sensor_yaml.find_last_of('/'));
  cam.name = dir.substr(dir.find_last_of('/') + 1);
  cam.model = makeKalibrCamera(n["camera_model"].as<std::string>(), distortion,
                               n["intrinsics"].as<std::vector<double>>(),
                               n["distortion_coefficients"].as<std::vector<double>>(), res.at(0),
                               res.at(1));
  cam.model_name = n["camera_model"].as<std::string>() + "-" + distortion;
  const auto data = n["T_BS"]["data"].as<std::vector<double>>();
  Eigen::Matrix4d T;
  for (int i = 0; i < 16; ++i) T(i / 4, i % 4) = data.at(i);
  cam.T_IC = Sophus::SE3d(Sophus::makeRotationMatrix(T.topLeftCorner<3, 3>()), T.topRightCorner<3, 1>());
  cam.has_T_IC = true;
  return cam;
}

KalibrCamera loadSequenceCamera(const std::string& euroc_dir, const std::string& cam, const std::string& camchain) {
  if (camchain.empty()) return loadEurocCamera(euroc_dir + "/mav0/" + cam + "/sensor.yaml");
  for (const auto& c : loadKalibrCamchain(camchain)) {
    if (c.name == cam) return c;
  }
  throw std::runtime_error(cam + " not in " + camchain);
}

ImuNoise loadKalibrImu(const std::string& path) {
  // Flat (EuRoC sensor.yaml) or nested under imu0 (Kalibr imu-*.yaml results).
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node n = root["imu0"] ? root["imu0"] : root;
  ImuNoise noise;
  noise.acc_noise = n["accelerometer_noise_density"].as<double>();
  noise.acc_bias_walk = n["accelerometer_random_walk"].as<double>();
  noise.gyro_noise = n["gyroscope_noise_density"].as<double>();
  noise.gyro_bias_walk = n["gyroscope_random_walk"].as<double>();
  return noise;
}

}  // namespace msckf::calib
