#include "msckf/init/featureless_init.h"

#include "msckf/imu/preintegration.h"

#include <Eigen/Dense>

#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>

namespace msckf {

namespace {

// Specific force excitation: the spread of the accelerometer reading rotated
// into {I0} (gravity is constant there, so only accelerations vary it), as the
// square root of the trace of its sample covariance.
double accExcitation(const std::vector<ImuSample>& imu, double t0, double t1, const Eigen::Vector3d& gyro_bias,
                     const Eigen::Vector3d& acc_bias) {
  Sophus::SO3d R;  // R_I0_Ik
  std::vector<Eigen::Vector3d> f;
  const ImuSample* prev = nullptr;
  for (const ImuSample& s : imu) {
    if (s.t < t0 || s.t > t1) continue;
    if (prev) R = R * Sophus::SO3d::exp((0.5 * (prev->gyro + s.gyro) - gyro_bias) * (s.t - prev->t));
    f.push_back(R * (s.acc - acc_bias));
    prev = &s;
  }
  if (f.size() < 2) return 0.0;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& x : f) mean += x;
  mean /= static_cast<double>(f.size());
  double var = 0.0;
  for (const auto& x : f) var += (x - mean).squaredNorm();
  return std::sqrt(var / static_cast<double>(f.size() - 1));
}

}  // namespace

std::vector<int> selectKeyframes(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                 const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias, int max_keyframes) {
  const int n = static_cast<int>(frames.size());
  std::vector<int> all(n);
  for (int k = 0; k < n; ++k) all[k] = k;
  if (n <= max_keyframes) return all;
  // Rotation of each frame's IMU relative to the first, from the gyro.
  std::vector<Sophus::SO3d> R(n);  // R_I0_Ik
  ImuPreintegration pre(gyro_bias, Eigen::Vector3d::Zero(), ImuNoise{});
  for (int k = 1; k < n; ++k) {
    pre.integrateRange(imu, frames[k - 1].t, frames[k].t);
    R[k] = pre.deltaR();
  }
  // Mean parallax between consecutive frames, accumulated.
  std::vector<double> cum(n, 0.0);
  for (int k = 1; k < n; ++k) {
    std::map<int64_t, Eigen::Vector3d> prev;
    for (const auto& [id, b] : frames[k - 1].bearings) prev[id] = R[k - 1] * T_IC.so3() * b;
    double sum = 0.0;
    int count = 0;
    for (const auto& [id, b] : frames[k].bearings) {
      const auto it = prev.find(id);
      if (it == prev.end()) continue;
      const Eigen::Vector3d bk = R[k] * T_IC.so3() * b;
      sum += std::atan2(bk.cross(it->second).norm(), bk.dot(it->second));
      ++count;
    }
    cum[k] = cum[k - 1] + (count ? sum / count : 0.0);
  }
  // First and last, and the frames closest to evenly spaced parallax between.
  std::vector<int> kf{0};
  for (int i = 1; i + 1 < max_keyframes; ++i) {
    const double target = cum.back() * i / (max_keyframes - 1);
    int best = kf.back() + 1;
    for (int k = kf.back() + 1; k < n - 1; ++k) {
      if (std::abs(cum[k] - target) < std::abs(cum[best] - target)) best = k;
    }
    if (best < n - 1) kf.push_back(best);
  }
  kf.push_back(n - 1);
  return kf;
}

FeaturelessInitResult featurelessInit(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                      const Sophus::SE3d& T_IC, const Eigen::Vector3d& gyro_bias,
                                      const Eigen::Vector3d& acc_bias, const ImuNoise& noise,
                                      const FeaturelessInitConfig& config) {
  FeaturelessInitResult res;
  const size_t n = frames.size();
  if (n < 2) throw std::invalid_argument("featurelessInit: needs at least two frames");
  const double t0 = frames.front().t;
  if (imu.empty() || imu.front().t > t0 || imu.back().t < frames.back().t) {
    res.reason = "imu";
    return res;
  }

  // IMU-induced motion of every frame relative to {I0}.
  std::vector<Sophus::SO3d> dR(n);
  std::vector<Eigen::Vector3d> alpha(n, Eigen::Vector3d::Zero());
  std::vector<double> dT(n, 0.0);
  std::vector<double> var_alpha(n, 0.0);  // mean per-axis variance of alpha_k
  ImuPreintegration pre(gyro_bias, acc_bias, noise);
  for (size_t k = 1; k < n; ++k) {
    if (!(frames[k].t > frames[k - 1].t)) throw std::invalid_argument("featurelessInit: frames out of order");
    pre.integrateRange(imu, frames[k - 1].t, frames[k].t);
    dR[k] = pre.deltaR();
    alpha[k] = pre.deltaP();
    dT[k] = pre.deltaTime();
    var_alpha[k] = pre.covariance().block<3, 3>(3, 3).trace() / 3.0;
  }

  // Bearings rotated into {I0}, by feature id.
  const Sophus::SO3d R_IC = T_IC.so3();
  const Eigen::Vector3d& p_IC = T_IC.translation();
  std::vector<std::map<int64_t, Eigen::Vector3d>> b(n);
  for (size_t k = 0; k < n; ++k) {
    const Sophus::SO3d R = dR[k] * R_IC;
    for (const auto& [id, bc] : frames[k].bearings) b[k][id] = R * bc;
  }

  // Two equations e^T A x = e^T rhs per usable pair, x = (v, g).
  std::vector<Eigen::Matrix<double, 2, 6>> rows;
  std::vector<Eigen::Vector2d> rhs;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix<double, 3, 6>>> pair_sys;  // (rb, A), for the scale
  double var_imu = 0.0;  // mean over pairs of the IMU's variance of alpha_k - alpha_j
  // Per usable pair, for the relative-scale equations: its index by (j, k)
  // and its translation direction (sign from positive depths).
  std::map<std::pair<size_t, size_t>, size_t> pair_index;
  std::vector<Eigen::Vector3d> pair_t;
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = j + 1; k < n; ++k) {
      Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
      int common = 0;
      for (const auto& [id, bj] : b[j]) {
        const auto it = b[k].find(id);
        if (it == b[k].end()) continue;
        const Eigen::Vector3d nrm = bj.cross(it->second);
        M += nrm * nrm.transpose();
        ++common;
      }
      if (common < config.min_common_features) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);  // ascending eigenvalues
      Eigen::Matrix<double, 3, 2> e;
      e << es.eigenvectors().col(1), es.eigenvectors().col(2);
      // s t = A x + rb, with A = [(dT_k - dT_j) I, 1/2 (dT_k^2 - dT_j^2) I]
      // (the paper's eq. 60 writes the same block as -A).
      Eigen::Matrix<double, 3, 6> A;
      A << (dT[k] - dT[j]) * Eigen::Matrix3d::Identity(),
          0.5 * (dT[k] * dT[k] - dT[j] * dT[j]) * Eigen::Matrix3d::Identity();
      const Eigen::Vector3d rb = alpha[k] - alpha[j] + (dR[k].matrix() - dR[j].matrix()) * p_IC;
      rows.push_back(e.transpose() * A);
      rhs.push_back(-e.transpose() * rb);
      pair_sys.push_back({rb, A});
      var_imu += var_alpha[k] + var_alpha[j];
      pair_index[{j, k}] = pair_t.size();
      pair_t.push_back(es.eigenvectors().col(0));
      res.pair_frames.emplace_back(static_cast<int>(j), static_cast<int>(k));
      res.pair_eigen_ratio.push_back(es.eigenvalues()(0) / std::max(es.eigenvalues()(1), 1e-300));
    }
  }
  res.pairs = static_cast<int>(rows.size());
  std::vector<Eigen::Matrix<double, 1, 6>> trows;
  std::vector<double> trhs;
  if (config.triplet_scale) {
    const double min_par = config.triplet_min_parallax_deg * M_PI / 180.0;
    // Depth at frame i per unit baseline t (from i to j), for bearings bi, bj
    // (both in {I0}): least squares of mu bi - lambda bj = t; nullopt below
    // the parallax threshold.
    auto unitDepth = [&](const Eigen::Vector3d& bi, const Eigen::Vector3d& bj,
                         const Eigen::Vector3d& t) -> std::optional<double> {
      if (std::atan2(bi.cross(bj).norm(), bi.dot(bj)) < min_par) return std::nullopt;
      Eigen::Matrix<double, 3, 2> M;
      M << bi, -bj;
      return (M.transpose() * M).ldlt().solve(M.transpose() * t)(0);
    };
    // Signs: the median unit depth of each pair's features must be positive.
    for (const auto& [jk, idx] : pair_index) {
      std::vector<double> mu;
      for (const auto& [id, bj] : b[jk.first]) {
        const auto it = b[jk.second].find(id);
        if (it == b[jk.second].end()) continue;
        if (const auto m = unitDepth(bj, it->second, pair_t[idx])) mu.push_back(*m);
      }
      if (mu.empty()) continue;
      std::nth_element(mu.begin(), mu.begin() + mu.size() / 2, mu.end());
      if (mu[mu.size() / 2] < 0.0) pair_t[idx] = -pair_t[idx];
    }
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        const auto ij = pair_index.find({i, j});
        if (ij == pair_index.end()) continue;
        for (size_t k = j + 1; k < n; ++k) {
          const auto ik = pair_index.find({i, k});
          if (ik == pair_index.end()) continue;
          const Eigen::Vector3d &tij = pair_t[ij->second], &tik = pair_t[ik->second];
          std::vector<double> ratio;
          for (const auto& [id, bi] : b[i]) {
            const auto itj = b[j].find(id), itk = b[k].find(id);
            if (itj == b[j].end() || itk == b[k].end()) continue;
            const auto mij = unitDepth(bi, itj->second, tij), mik = unitDepth(bi, itk->second, tik);
            if (mij && mik && *mij > 0.0 && *mik > 0.0) ratio.push_back(*mik / *mij);
          }
          if (static_cast<int>(ratio.size()) < config.triplet_min_features) continue;
          std::nth_element(ratio.begin(), ratio.begin() + ratio.size() / 2, ratio.end());
          const double rho = ratio[ratio.size() / 2];
          const double w = config.triplet_weight / std::sqrt(1.0 + rho * rho);
          const auto& [rb_ij, A_ij] = pair_sys[ij->second];
          const auto& [rb_ik, A_ik] = pair_sys[ik->second];
          trows.push_back(w * (tij.transpose() * A_ij - rho * tik.transpose() * A_ik));
          trhs.push_back(-w * (tij.dot(rb_ij) - rho * tik.dot(rb_ik)));
        }
      }
    }
  }
  res.triplets = static_cast<int>(trows.size());
  res.acc_excitation = accExcitation(imu, t0, frames.back().t, gyro_bias, acc_bias);
  // Six unknowns (v_I0, g_I0): fewer rows cannot determine them.
  if (res.pairs < config.min_pairs || 2 * res.pairs + res.triplets < 6) {
    res.reason = "pairs";
    return res;
  }

  const int m = 2 * res.pairs + res.triplets;
  Eigen::MatrixXd C(m, 6);
  Eigen::VectorXd d(m);
  for (int i = 0; i < res.pairs; ++i) {
    C.middleRows<2>(2 * i) = rows[i];
    d.segment<2>(2 * i) = rhs[i];
  }
  for (int i = 0; i < res.triplets; ++i) {
    C.row(2 * res.pairs + i) = trows[i];
    d(2 * res.pairs + i) = trhs[i];
  }

  // Linear least squares on the column-scaled system; its smallest singular
  // value measures how well (v, g) are constrained.
  Eigen::VectorXd scale = C.colwise().norm().cwiseMax(1e-12).cwiseInverse();
  const Eigen::MatrixXd Cs = C * scale.asDiagonal();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cs, Eigen::ComputeThinU | Eigen::ComputeThinV);
  res.singular_value = svd.singularValues()(5);
  Eigen::Matrix<double, 6, 1> x = scale.asDiagonal() * svd.solve(d);

  // Gauss-Newton with |g| fixed: g = |g| u, u on the unit sphere.
  Eigen::Vector3d u = x.tail<3>();
  if (u.norm() < 1e-6) {
    res.reason = "conditioning";
    return res;
  }
  u.normalize();
  Eigen::Vector3d v = x.head<3>();
  const double gn = config.gravity_norm;
  // Tangent basis of the sphere at u, and the Jacobian of the residual
  // C (v, gn u) - d in (v, tangent step).
  Eigen::Matrix<double, 3, 2> B;
  Eigen::MatrixXd J(m, 5);
  auto linearize = [&]() {
    Eigen::Vector3d a = std::abs(u.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    B.col(0) = u.cross(a).normalized();
    B.col(1) = u.cross(B.col(0));
    J.leftCols<3>() = C.leftCols<3>();
    J.rightCols<2>() = C.rightCols<3>() * gn * B;
  };
  auto residual = [&]() -> Eigen::VectorXd {
    Eigen::Matrix<double, 6, 1> xv;
    xv << v, gn * u;
    return C * xv - d;
  };
  for (int it = 0; it < config.gravity_iterations; ++it) {
    linearize();
    const Eigen::Matrix<double, 5, 1> dx = J.colPivHouseholderQr().solve(-residual());
    v += dx.head<3>();
    u = (u + B * dx.tail<2>()).normalized();
    if (dx.norm() < 1e-12) break;
  }
  linearize();
  {
    const Eigen::VectorXd r = residual();
    const double var_fit = m > 5 ? r.squaredNorm() / (m - 5) : 0.0;
    const double var = std::max(var_fit, var_imu / res.pairs);
    const Eigen::Matrix<double, 5, 5> H = J.transpose() * J;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 5, 5>> es(H);
    // Pseudo-inverse with unconstrained directions at a huge variance.
    const double floor = es.eigenvalues().maxCoeff() * 1e-14;
    Eigen::Matrix<double, 5, 1> inv;
    for (int i = 0; i < 5; ++i) inv(i) = es.eigenvalues()(i) > floor ? 1.0 / es.eigenvalues()(i) : 1e12;
    res.covariance = var * es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
    res.gravity_basis = B;
    res.has_covariance = true;
  }
  res.v_I0 = v;
  res.g_I0 = gn * u;
  x << v, res.g_I0;
  for (const auto& [rb, A] : pair_sys) res.pair_scale.push_back((A * x + rb).norm());

  if (res.singular_value < config.min_singular_value) {
    res.reason = "conditioning";
  } else if (res.acc_excitation < config.min_acc_excitation) {
    res.reason = "excitation";
  } else {
    res.ok = true;
  }
  return res;
}

}  // namespace msckf
