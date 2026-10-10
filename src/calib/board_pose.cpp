#include "msckf/calib/board_pose.h"

#include <Eigen/Cholesky>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <cmath>

namespace msckf::calib {

namespace {

// Stacked residuals and the Jacobian w.r.t. a left perturbation
// T <- Exp(delta) T, delta = [rho; phi] (Sophus order). Returns false if any
// corner falls outside the model's validity region.
bool linearize(const CameraModel& camera, const AprilGrid& grid, const GridObservation& obs,
               const Sophus::SE3d& T, Eigen::VectorXd* r, Eigen::MatrixXd* J) {
  const int n = static_cast<int>(obs.size());
  r->resize(2 * n);
  if (J) J->resize(2 * n, 6);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d p_c = T * grid.corner(obs.ids[i]);
    Eigen::Vector2d uv;
    Matrix23d Jp;
    if (!camera.project(p_c, uv, J ? &Jp : nullptr)) return false;
    r->segment<2>(2 * i) = obs.pixels[i] - uv;
    if (J) {
      J->block<2, 3>(2 * i, 0) = Jp;
      J->block<2, 3>(2 * i, 3) = -Jp * Sophus::SO3d::hat(p_c);
    }
  }
  return true;
}

bool initialPose(const CameraModel& camera, const AprilGrid& grid, const GridObservation& obs,
                 Sophus::SE3d* T) {
  std::vector<cv::Point3d> object;
  std::vector<cv::Point2d> normalized;
  for (size_t i = 0; i < obs.size(); ++i) {
    Eigen::Vector3d b;
    if (!camera.unproject(obs.pixels[i], b) || b.z() < 0.1) continue;  // within ~84 deg
    const Eigen::Vector3d P = grid.corner(obs.ids[i]);
    object.emplace_back(P.x(), P.y(), P.z());
    normalized.emplace_back(b.x() / b.z(), b.y() / b.z());
  }
  if (object.size() < 4) return false;
  cv::Mat rvec, tvec;
  if (!cv::solvePnP(object, normalized, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec, tvec, false,
                    cv::SOLVEPNP_IPPE)) {
    return false;
  }
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  Eigen::Matrix3d Re;
  Eigen::Vector3d te;
  cv::cv2eigen(R, Re);
  cv::cv2eigen(tvec, te);
  *T = Sophus::SE3d(Sophus::makeRotationMatrix(Re), te);
  return true;
}

}  // namespace

BoardPoseResult fitBoardPose(const CameraModel& camera, const AprilGrid& grid,
                             const GridObservation& obs, int max_iterations) {
  BoardPoseResult res;
  Sophus::SE3d T;
  if (obs.size() < 4 || !initialPose(camera, grid, obs, &T)) return res;

  Eigen::VectorXd r;
  Eigen::MatrixXd J;
  if (!linearize(camera, grid, obs, T, &r, &J)) return res;
  double cost = r.squaredNorm();
  double lambda = 1e-4;
  bool converged = false;
  while (!converged && res.iterations < max_iterations) {
    ++res.iterations;
    const Eigen::Matrix<double, 6, 6> H = J.transpose() * J;
    const Eigen::Matrix<double, 6, 1> g = J.transpose() * r;
    bool improved = false;
    while (lambda < 1e8) {
      Eigen::Matrix<double, 6, 6> A = H;
      A.diagonal() += lambda * H.diagonal().cwiseMax(1e-9);
      // r = obs - proj(T), J = d proj / d delta, so the step solves A d = J^T r.
      const Eigen::Matrix<double, 6, 1> delta = A.ldlt().solve(g);
      const Sophus::SE3d T_new = Sophus::SE3d::exp(delta) * T;
      Eigen::VectorXd r_new;
      if (linearize(camera, grid, obs, T_new, &r_new, nullptr) && r_new.squaredNorm() < cost) {
        const double rel = (cost - r_new.squaredNorm()) / std::max(cost, 1e-30);
        T = T_new;
        cost = r_new.squaredNorm();
        lambda = std::max(lambda / 10.0, 1e-10);
        linearize(camera, grid, obs, T, &r, &J);
        improved = true;
        converged = rel < 1e-12 || delta.norm() < 1e-12;
        break;
      }
      lambda *= 10.0;
    }
    if (!improved) break;
  }

  res.T_cam_board = T;
  res.residuals.resize(obs.size());
  for (size_t i = 0; i < obs.size(); ++i) res.residuals[i] = r.segment<2>(2 * i);
  res.rms = std::sqrt(cost / static_cast<double>(obs.size()));
  res.ok = true;
  return res;
}

}  // namespace msckf::calib
