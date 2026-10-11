#include "msckf/init/featureless_init.h"

#include "msckf/imu/preintegration.h"

#include <Eigen/Dense>

#include <cmath>
#include <map>
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
  ImuPreintegration pre(gyro_bias, acc_bias, noise);
  for (size_t k = 1; k < n; ++k) {
    if (!(frames[k].t > frames[k - 1].t)) throw std::invalid_argument("featurelessInit: frames out of order");
    pre.integrateRange(imu, frames[k - 1].t, frames[k].t);
    dR[k] = pre.deltaR();
    alpha[k] = pre.deltaP();
    dT[k] = pre.deltaTime();
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
      res.pair_frames.emplace_back(static_cast<int>(j), static_cast<int>(k));
      res.pair_eigen_ratio.push_back(es.eigenvalues()(0) / std::max(es.eigenvalues()(1), 1e-300));
    }
  }
  res.pairs = static_cast<int>(rows.size());
  res.acc_excitation = accExcitation(imu, t0, frames.back().t, gyro_bias, acc_bias);
  if (res.pairs < config.min_pairs) {
    res.reason = "pairs";
    return res;
  }

  const int m = 2 * res.pairs;
  Eigen::MatrixXd C(m, 6);
  Eigen::VectorXd d(m);
  for (int i = 0; i < res.pairs; ++i) {
    C.middleRows<2>(2 * i) = rows[i];
    d.segment<2>(2 * i) = rhs[i];
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
  for (int it = 0; it < config.gravity_iterations; ++it) {
    // Tangent basis of the sphere at u.
    Eigen::Vector3d a = std::abs(u.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    Eigen::Matrix<double, 3, 2> B;
    B.col(0) = u.cross(a).normalized();
    B.col(1) = u.cross(B.col(0));
    Eigen::Matrix<double, 6, 1> xv;
    xv << v, gn * u;
    const Eigen::VectorXd r = C * xv - d;
    Eigen::MatrixXd J(m, 5);
    J.leftCols<3>() = C.leftCols<3>();
    J.rightCols<2>() = C.rightCols<3>() * gn * B;
    const Eigen::Matrix<double, 5, 1> dx = J.colPivHouseholderQr().solve(-r);
    v += dx.head<3>();
    u = (u + B * dx.tail<2>()).normalized();
    if (dx.norm() < 1e-12) break;
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
