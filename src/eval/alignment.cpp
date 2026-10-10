#include "msckf/eval/alignment.h"

#include <Eigen/SVD>

#include <cmath>
#include <stdexcept>

namespace msckf::eval {

AlignmentType parseAlignment(const std::string& name) {
  if (name == "none") return AlignmentType::kNone;
  if (name == "posyaw") return AlignmentType::kPosYaw;
  if (name == "se3") return AlignmentType::kSE3;
  if (name == "sim3") return AlignmentType::kSim3;
  throw std::invalid_argument("unknown alignment '" + name + "' (none, posyaw, se3, sim3)");
}

std::string toString(AlignmentType type) {
  switch (type) {
    case AlignmentType::kNone: return "none";
    case AlignmentType::kPosYaw: return "posyaw";
    case AlignmentType::kSE3: return "se3";
    case AlignmentType::kSim3: return "sim3";
  }
  return "unknown";
}

Alignment align(const std::vector<Eigen::Vector3d>& est, const std::vector<Eigen::Vector3d>& gt,
                AlignmentType type) {
  if (est.size() != gt.size()) throw std::invalid_argument("align: size mismatch");
  Alignment out;
  if (type == AlignmentType::kNone || est.empty()) return out;
  if (est.size() < 3 && type != AlignmentType::kPosYaw) {
    throw std::invalid_argument("align: need at least 3 points");
  }

  const double n = static_cast<double>(est.size());
  Eigen::Vector3d mu_e = Eigen::Vector3d::Zero(), mu_g = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < est.size(); ++i) {
    mu_e += est[i];
    mu_g += gt[i];
  }
  mu_e /= n;
  mu_g /= n;

  if (type == AlignmentType::kPosYaw) {
    // min_theta sum |Rz(theta) e_i - g_i|^2 over centered points: only the
    // horizontal components depend on theta, giving theta = atan2(S_cross, S_dot).
    double s_dot = 0.0, s_cross = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
      const Eigen::Vector3d e = est[i] - mu_e, g = gt[i] - mu_g;
      s_dot += e.x() * g.x() + e.y() * g.y();
      s_cross += e.x() * g.y() - e.y() * g.x();
    }
    out.R = Sophus::SO3d::rotZ(std::atan2(s_cross, s_dot));
    out.t = mu_g - out.R * mu_e;
    return out;
  }

  // Umeyama: Sigma = 1/n sum (g - mu_g)(e - mu_e)^T = U D V^T, R = U S V^T.
  Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
  double var_e = 0.0;
  for (size_t i = 0; i < est.size(); ++i) {
    const Eigen::Vector3d e = est[i] - mu_e;
    sigma += (gt[i] - mu_g) * e.transpose();
    var_e += e.squaredNorm();
  }
  sigma /= n;
  var_e /= n;
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) S(2, 2) = -1.0;
  out.R = Sophus::SO3d(svd.matrixU() * S * svd.matrixV().transpose());
  if (type == AlignmentType::kSim3) {
    out.scale = (svd.singularValues().asDiagonal() * S).trace() / var_e;
  }
  out.t = mu_g - out.scale * (out.R * mu_e);
  return out;
}

}  // namespace msckf::eval
