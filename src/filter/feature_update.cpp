#include "msckf/filter/feature_update.h"

#include "msckf/eval/nees.h"
#include "msckf/geometry/triangulation.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

namespace msckf::filter {

namespace {

// Largest angle between the rays from the cameras to the point, deg.
double maxRayAngleDeg(const std::vector<Sophus::SE3d>& T_GC, const Eigen::Vector3d& p) {
  double best = 0.0;
  for (size_t i = 0; i < T_GC.size(); ++i) {
    for (size_t j = i + 1; j < T_GC.size(); ++j) {
      const Eigen::Vector3d a = p - T_GC[i].translation(), b = p - T_GC[j].translation();
      best = std::max(best, std::atan2(a.cross(b).norm(), a.dot(b)));
    }
  }
  return best * 180.0 / M_PI;
}

}  // namespace

double chi2Quantile(double probability, int dof) {
  static std::mutex mutex;
  static std::map<std::pair<double, int>, double> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_pair(probability, dof);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, eval::chiSquareQuantile(probability, dof)).first;
  return it->second;
}

bool linearizeFeature(const FilterState& s, const CameraModel& camera, const FeatureTrack& track,
                      const std::vector<size_t>& clone_of_obs, const Eigen::Vector3d& p_f, FeatureJacobian* out,
                      const Eigen::Vector3d* p_f_lin) {
  const int m = static_cast<int>(clone_of_obs.size());
  out->H_x = Eigen::MatrixXd::Zero(2 * m, s.dim());
  out->H_f = Eigen::MatrixXd::Zero(2 * m, 3);
  out->r = Eigen::VectorXd::Zero(2 * m);
  const Eigen::Matrix3d R_CI = s.R_IC.inverse().matrix();
  const bool fej = s.options().fej;
  for (int j = 0; j < m; ++j) {
    const Clone& c = s.clones()[clone_of_obs[j]];
    // Residual at the current estimates.
    const Eigen::Vector3d p_C = R_CI * (c.R_GI.inverse() * (p_f - c.p_GI) - s.p_IC);
    Eigen::Vector2d uv;
    if (!camera.project(p_C, uv)) return false;
    out->r.segment<2>(2 * j) = track.uv[j] - uv;
    // Jacobians at the linearization point: with FEJ the clone's first
    // estimates (see Clone).
    const Sophus::SO3d& R_GI = fej ? c.R_GI_fej : c.R_GI;
    const Eigen::Vector3d& p_GI = fej ? c.p_GI_fej : c.p_GI;
    const Eigen::Vector3d p_I_lin = R_GI.inverse() * ((p_f_lin ? *p_f_lin : p_f) - p_GI);
    const Eigen::Vector3d p_C_lin = R_CI * (p_I_lin - s.p_IC);
    Eigen::Vector2d uv_lin;
    Matrix23d J;
    if (!camera.project(p_C_lin, uv_lin, &J)) return false;
    const Eigen::Matrix3d R_IG = R_GI.inverse().matrix();
    const int ci = s.cloneIndex(clone_of_obs[j]);
    out->H_x.block<2, 3>(2 * j, ci) = J * R_CI * Sophus::SO3d::hat(p_I_lin);
    out->H_x.block<2, 3>(2 * j, ci + 3) = -J * R_CI * R_IG;
    out->H_f.block<2, 3>(2 * j, 0) = J * R_CI * R_IG;
    if (const int e = s.extrinsicIndex(); e >= 0) {
      out->H_x.block<2, 3>(2 * j, e) = J * Sophus::SO3d::hat(p_C_lin);
      out->H_x.block<2, 3>(2 * j, e + 3) = -J * R_CI;
    }
  }
  return true;
}

UpdateStats featureUpdate(FilterState& s, const CameraModel& camera, const std::vector<FeatureTrack>& tracks,
                          const UpdateOptions& options) {
  UpdateStats stats;
  std::map<double, size_t> clone_at;
  for (size_t i = 0; i < s.clones().size(); ++i) clone_at[s.clones()[i].t_cam] = i;
  const double sigma2 = options.pixel_noise * options.pixel_noise;
  const Sophus::SE3d T_IC(s.R_IC, s.p_IC);

  std::vector<Eigen::MatrixXd> Hs;
  std::vector<Eigen::VectorXd> rs;
  for (const FeatureTrack& track : tracks) {
    // Observations that have a clone.
    FeatureTrack obs;
    std::vector<size_t> clone_of_obs;
    std::vector<Sophus::SE3d> T_GC;
    for (size_t k = 0; k < track.t_cam.size(); ++k) {
      const auto it = clone_at.find(track.t_cam[k]);
      if (it == clone_at.end()) continue;
      const Clone& c = s.clones()[it->second];
      obs.t_cam.push_back(track.t_cam[k]);
      obs.uv.push_back(track.uv[k]);
      obs.bearing.push_back(track.bearing[k]);
      clone_of_obs.push_back(it->second);
      T_GC.push_back(Sophus::SE3d(c.R_GI, c.p_GI) * T_IC);
    }
    if (static_cast<int>(clone_of_obs.size()) < options.min_observations) continue;
    ++stats.features;

    // One estimate of the point, triangulated from the current clones, for
    // the residual and every Jacobian (MSCKF 2.0).
    Eigen::Vector3d p_f, p_f_lin, p_true;
    double parallax = 0.0;
    const bool have_true = options.landmark_oracle && options.landmark_oracle(track.id, &p_true);
    const bool ok = triangulate(T_GC, obs.bearing, &p_f, &parallax);
    if (have_true && options.oracle_selection) parallax = maxRayAngleDeg(T_GC, p_true);
    if (!ok || parallax < options.min_parallax_deg) {
      ++stats.rejected_triangulation;
      continue;
    }
    p_f_lin = p_f;
    if (have_true && options.oracle_jacobians) p_f_lin = p_true;
    FeatureJacobian fj;
    if (!linearizeFeature(s, camera, obs, clone_of_obs, p_f, &fj, &p_f_lin)) {
      ++stats.rejected_triangulation;
      continue;
    }

    // Left nullspace of H_f: the last 2M - 3 rows of Q^T.
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(fj.H_f);
    const Eigen::MatrixXd Qt = qr.householderQ().transpose();
    const int rows = static_cast<int>(fj.r.size()) - 3;
    const Eigen::MatrixXd H_o = (Qt * fj.H_x).bottomRows(rows);
    const Eigen::VectorXd r_o = (Qt * fj.r).tail(rows);

    const Eigen::MatrixXd S = H_o * s.P * H_o.transpose() + sigma2 * Eigen::MatrixXd::Identity(rows, rows);
    const double gamma = r_o.dot(S.ldlt().solve(r_o));
    if (gamma > chi2Quantile(options.chi2_probability, rows)) {
      ++stats.rejected_chi2;
      continue;
    }
    Hs.push_back(H_o);
    rs.push_back(r_o);
    ++stats.used;
    stats.rows += rows;
    stats.nis += gamma;
    if (have_true) {
      const Sophus::SE3d T_CG = T_GC.front().inverse();
      stats.depth_log_ratio.push_back(std::log((T_CG * p_f).norm() / (T_CG * p_true).norm()));
    }
  }
  if (Hs.empty()) return stats;

  Eigen::MatrixXd H(stats.rows, s.dim());
  Eigen::VectorXd r(stats.rows);
  for (size_t i = 0, row = 0; i < Hs.size(); row += Hs[i].rows(), ++i) {
    H.middleRows(row, Hs[i].rows()) = Hs[i];
    r.segment(row, rs[i].size()) = rs[i];
  }
  // Compress: H = Q [T; 0], keep T and the matching part of Q^T r.
  if (H.rows() > H.cols()) {
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(H);
    const Eigen::VectorXd Qtr = qr.householderQ().transpose() * r;
    const int n = static_cast<int>(H.cols());
    H = qr.matrixQR().topRows(n).triangularView<Eigen::Upper>();
    r = Qtr.head(n);
  }

  const Eigen::MatrixXd PHt = s.P * H.transpose();
  const Eigen::MatrixXd S = H * PHt + sigma2 * Eigen::MatrixXd::Identity(H.rows(), H.rows());
  const Eigen::MatrixXd K = S.ldlt().solve(PHt.transpose()).transpose();
  const Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(s.dim(), s.dim()) - K * H;
  s.P = IKH * s.P * IKH.transpose() + sigma2 * K * K.transpose();
  s.symmetrize();
  s.inject(K * r);
  return stats;
}

}  // namespace msckf::filter
