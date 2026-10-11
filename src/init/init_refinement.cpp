#include "msckf/init/init_refinement.h"

#include "msckf/core/constants.h"
#include "msckf/eval/nees.h"
#include "msckf/imu/preintegration.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <map>

namespace msckf {

namespace {

using Matrix32d = Eigen::Matrix<double, 3, 2>;

// Orthonormal basis of the plane orthogonal to the unit vector b.
Matrix32d tangentBasis(const Eigen::Vector3d& b) {
  const Eigen::Vector3d a = std::abs(b.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  Matrix32d B;
  B.col(0) = b.cross(a).normalized();
  B.col(1) = b.cross(B.col(0));
  return B;
}

// Point from camera poses T_GC and unit bearings in each camera: least
// squares on the perpendicular distances to the rays, then Gauss-Newton on
// the tangent-plane bearing residuals. False for a point behind a camera or
// rays too close to parallel.
bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, Eigen::Vector3d* X,
                 double* parallax_deg) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> d(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    d[i] = T_GC[i].so3() * b[i];
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d[i] * d[i].transpose();
    A += P;
    rhs += P * T_GC[i].translation();
  }
  double max_angle = 0.0;
  for (size_t i = 0; i < d.size(); ++i) {
    for (size_t j = i + 1; j < d.size(); ++j) {
      max_angle = std::max(max_angle, std::atan2(d[i].cross(d[j]).norm(), d[i].dot(d[j])));
    }
  }
  *parallax_deg = max_angle * 180.0 / M_PI;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.eigenvalues()(0) < 1e-9 * es.eigenvalues()(2)) return false;
  *X = A.ldlt().solve(rhs);
  for (int it = 0; it < 5; ++it) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < b.size(); ++i) {
      const Eigen::Vector3d pc = T_GC[i].inverse() * *X;
      const double r = pc.norm();
      const Eigen::Vector3d n = pc / r;
      const Matrix32d B = tangentBasis(b[i]);
      const Eigen::Matrix<double, 2, 3> J =
          B.transpose() * (Eigen::Matrix3d::Identity() - n * n.transpose()) / r * T_GC[i].so3().inverse().matrix();
      const Eigen::Vector2d res = -B.transpose() * n;
      H += J.transpose() * J;
      g += J.transpose() * res;
    }
    const Eigen::Vector3d dX = H.ldlt().solve(g);
    *X += dX;
    if (dX.norm() < 1e-9 * (1.0 + X->norm())) break;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    if ((T_GC[i].inverse() * *X).dot(b[i]) <= 0.0) return false;
  }
  return true;
}

Eigen::Matrix3d hat(const Eigen::Vector3d& v) { return Sophus::SO3d::hat(v); }

// A feature in the refinement's state: global position, observations (frame,
// bearing in the camera) and, per iteration, its information blocks.
struct Landmark {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  std::vector<std::pair<int, Eigen::Vector3d>> obs;
  bool inlier = true, used = false;
  Eigen::Matrix3d Lff, Lff_inv;
  Eigen::MatrixXd Lxf;  // n x 3
  Eigen::Vector3d gf;
};

}  // namespace

InitRefinementResult refineInit(const std::vector<ImuSample>& imu, const std::vector<InitFrame>& frames,
                                const Sophus::SE3d& T_IC, const FeaturelessInitResult& linear,
                                const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& acc_bias,
                                const ImuNoise& noise, const InitRefinementConfig& config) {
  InitRefinementResult res;
  const int N = static_cast<int>(frames.size());
  const int n = 9 * N + 6;
  const int kB = 9 * N;  // gyro bias at kB, accelerometer bias at kB + 3

  // Initial estimate: first clone from the linear solve, the rest propagated.
  std::vector<ImuPreintegration> pre;
  std::vector<NavState> x(N);
  x[0].R_GI = attitudeFromUp(-linear.g_I0.normalized());
  x[0].v_GI = x[0].R_GI * linear.v_I0;
  for (int k = 1; k < N; ++k) {
    pre.emplace_back(gyro_bias, acc_bias, noise);
    pre.back().integrateRange(imu, frames[k - 1].t, frames[k].t);
    x[k] = pre.back().predict(x[k - 1], gyro_bias, acc_bias);
  }
  Eigen::Vector3d bg = gyro_bias, ba = acc_bias;
  const NavState prior0 = x[0];

  // Prior information: first clone (loose tilt and velocity, tight yaw and
  // position: the gauge) and the biases (prior calibration).
  Eigen::MatrixXd L0 = Eigen::MatrixXd::Zero(n, n);
  {
    const Eigen::Matrix3d R0 = prior0.R_GI.matrix();
    const Eigen::Vector3d s(config.tilt_sigma, config.tilt_sigma, config.gauge_sigma);
    L0.block<3, 3>(0, 0) = R0.transpose() * s.cwiseAbs2().cwiseInverse().asDiagonal() * R0;  // dtheta in the IMU frame
    L0.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() / std::pow(config.gauge_sigma, 2);
    L0.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() / std::pow(config.velocity_sigma, 2);
    L0.block<3, 3>(kB, kB) = Eigen::Matrix3d::Identity() / std::pow(config.gyro_bias_sigma, 2);
    L0.block<3, 3>(kB + 3, kB + 3) = Eigen::Matrix3d::Identity() / std::pow(config.acc_bias_sigma, 2);
  }

  // Observations by feature: (frame, bearing in the camera).
  std::map<int64_t, std::vector<std::pair<int, Eigen::Vector3d>>> tracks;
  for (int k = 0; k < N; ++k) {
    for (const auto& [id, b] : frames[k].bearings) tracks[id].emplace_back(k, b);
  }

  const double var = config.bearing_sigma * config.bearing_sigma;
  const Eigen::Matrix3d R_IC = T_IC.so3().matrix();
  const Eigen::Vector3d& p_IC = T_IC.translation();
  Eigen::MatrixXd Lambda;
  Eigen::MatrixXd cov;  // of the previous iterate, for the chi-square gate
  // Landmarks: every feature seen in >= 2 frames with enough parallax,
  // triangulated once from the initial clones, then estimated in the state
  // (the paper's key features).
  std::map<int64_t, Landmark> landmarks;
  {
    std::vector<Sophus::SE3d> T_GC(N);
    for (int k = 0; k < N; ++k) T_GC[k] = Sophus::SE3d(x[k].R_GI, x[k].p_GI) * T_IC;
    for (const auto& [id, obs] : tracks) {
      if (obs.size() < 2) continue;
      std::vector<Sophus::SE3d> poses;
      std::vector<Eigen::Vector3d> bs;
      for (const auto& [k, bb] : obs) {
        poses.push_back(T_GC[k]);
        bs.push_back(bb);
      }
      Landmark lm;
      double parallax = 0.0;
      if (!triangulate(poses, bs, &lm.p, &parallax) || parallax < config.min_parallax_deg) continue;
      lm.obs = obs;
      landmarks[id] = std::move(lm);
    }
  }
  bool converged = false, gated = false, gate_now = false;
  for (int it = 0; it < config.max_iterations; ++it) {
    Lambda = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n);  // -J^T W r
    double cost = 0.0;
    int rows = 0;

    // Prior: residual = estimate (-) prior, Jacobian ~ identity.
    {
      Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
      e.segment<3>(0) = (prior0.R_GI.inverse() * x[0].R_GI).log();
      e.segment<3>(3) = x[0].p_GI - prior0.p_GI;
      e.segment<3>(6) = x[0].v_GI - prior0.v_GI;
      e.segment<3>(kB) = bg - gyro_bias;
      e.segment<3>(kB + 3) = ba - acc_bias;
      Lambda += L0;
      g -= L0 * e;
      cost += e.dot(L0 * e);
      rows += 15;
    }

    // IMU: preintegration between consecutive clones (Forster et al.),
    // relinearized at every iterate.
    for (int k = 1; k < N; ++k) {
      const ImuPreintegration& pk = pre[k - 1];
      const double T = pk.deltaTime();
      const NavState &a = x[k - 1], &b = x[k];
      const Eigen::Matrix3d Ra_t = a.R_GI.inverse().matrix();
      const Sophus::SO3d dR = pk.deltaR(bg);
      Eigen::Matrix<double, 9, 1> r;  // order (theta, p, v), as the preintegration covariance
      r.segment<3>(0) = (dR.inverse() * a.R_GI.inverse() * b.R_GI).log();
      const Eigen::Vector3d dp_w = b.p_GI - a.p_GI - a.v_GI * T - 0.5 * kGravity * T * T;
      const Eigen::Vector3d dv_w = b.v_GI - a.v_GI - kGravity * T;
      r.segment<3>(3) = Ra_t * dp_w - pk.deltaP(bg, ba);
      r.segment<3>(6) = Ra_t * dv_w - pk.deltaV(bg, ba);
      const Eigen::Matrix3d Jr_inv = so3RightJacobian(r.segment<3>(0)).inverse();
      Eigen::Matrix<double, 9, Eigen::Dynamic> J = Eigen::Matrix<double, 9, Eigen::Dynamic>::Zero(9, n);
      const int ia = 9 * (k - 1), ib = 9 * k;
      J.block<3, 3>(0, ia) = -Jr_inv * (b.R_GI.inverse() * a.R_GI).matrix();
      J.block<3, 3>(0, ib) = Jr_inv;
      J.block<3, 3>(0, kB) = -Jr_inv * Sophus::SO3d::exp(r.segment<3>(0)).inverse().matrix() *
                             so3RightJacobian(pk.dR_dbg() * (bg - pk.gyroBias())) * pk.dR_dbg();
      J.block<3, 3>(3, ia) = hat(Ra_t * dp_w);
      J.block<3, 3>(3, ia + 3) = -Ra_t;
      J.block<3, 3>(3, ia + 6) = -Ra_t * T;
      J.block<3, 3>(3, ib + 3) = Ra_t;
      J.block<3, 3>(3, kB) = -pk.dP_dbg();
      J.block<3, 3>(3, kB + 3) = -pk.dP_dba();
      J.block<3, 3>(6, ia) = hat(Ra_t * dv_w);
      J.block<3, 3>(6, ia + 6) = -Ra_t;
      J.block<3, 3>(6, ib + 6) = Ra_t;
      J.block<3, 3>(6, kB) = -pk.dV_dbg();
      J.block<3, 3>(6, kB + 3) = -pk.dV_dba();
      const Matrix9d W = pk.covariance().inverse();
      Lambda += J.transpose() * W * J;
      g -= J.transpose() * W * r;
      cost += r.dot(W * r);
      rows += 9;
    }

    // Vision: every landmark in the state, residuals on the tangent plane of
    // each observed bearing. Its blocks are kept apart (Lff, Lxf, gf) so it
    // can be eliminated by the Schur complement below.
    res.features = res.rejected = 0;
    for (auto& [id, lm] : landmarks) {
      lm.used = false;
      if (gated && !lm.inlier) {
        ++res.rejected;
        continue;
      }
      const int m = static_cast<int>(lm.obs.size());
      Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(2 * m, n);
      Eigen::MatrixXd Hf(2 * m, 3);
      Eigen::VectorXd r(2 * m);
      bool in_front = true;
      for (int o = 0; o < m; ++o) {
        const auto& [k, bb] = lm.obs[o];
        const Eigen::Matrix3d Rk = x[k].R_GI.matrix();
        const Eigen::Vector3d pI = Rk.transpose() * (lm.p - x[k].p_GI);
        const Eigen::Vector3d pc = R_IC.transpose() * (pI - p_IC);
        in_front &= pc.dot(bb) > 0.0;
        const double rng = pc.norm();
        const Eigen::Vector3d nc = pc / rng;
        const Matrix32d B = tangentBasis(bb);
        const Eigen::Matrix<double, 2, 3> Jn = B.transpose() * (Eigen::Matrix3d::Identity() - nc * nc.transpose()) / rng;
        r.segment<2>(2 * o) = B.transpose() * nc;  // predicted minus observed (B^T b = 0)
        Hx.block<2, 3>(2 * o, 9 * k) = Jn * R_IC.transpose() * hat(pI);
        Hx.block<2, 3>(2 * o, 9 * k + 3) = -Jn * R_IC.transpose() * Rk.transpose();
        Hf.block<2, 3>(2 * o, 0) = Jn * R_IC.transpose() * Rk.transpose();
      }
      if (!in_front) {
        lm.inlier = false;
        ++res.rejected;
        continue;
      }
      // Chi-square gate in two phases: iterate to convergence with every
      // landmark, decide once there (where inliers fit to the noise level),
      // then iterate again with the inliers; fixed after, so iterates cannot
      // oscillate by swapping landmarks in and out.
      if (gate_now) {
        lm.inlier = r.squaredNorm() / var <= eval::chiSquareQuantile(config.chi2_quantile, 2 * m - 3);
        if (!lm.inlier) {
          ++res.rejected;
          continue;
        }
      }
      lm.Lff = Hf.transpose() * Hf / var;
      lm.Lxf = Hx.transpose() * Hf / var;
      lm.gf = -Hf.transpose() * r / var;
      Lambda += Hx.transpose() * Hx / var;
      g -= Hx.transpose() * r / var;
      cost += r.squaredNorm() / var;
      rows += 2 * m;
      lm.used = true;
      ++res.features;
    }
    if (gate_now) {
      gated = true;
      gate_now = false;
    }
    if (res.features < 5) {
      res.reason = "features";
      return res;
    }

    // Gauss-Newton step with the landmarks eliminated (Schur complement on
    // their 3x3 blocks), then their own step from the back-substitution.
    Eigen::MatrixXd S = Lambda;
    Eigen::VectorXd b = g;
    for (auto& [id, lm] : landmarks) {
      if (!lm.used) continue;
      lm.Lff_inv = lm.Lff.inverse();
      S -= lm.Lxf * lm.Lff_inv * lm.Lxf.transpose();
      b -= lm.Lxf * lm.Lff_inv * lm.gf;
    }
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    const Eigen::VectorXd dx = ldlt.solve(b);
    cov = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    res.iterations = it + 1;
    res.cost = cost;
    res.dof = rows - n - 3 * res.features;
    double step = dx.cwiseAbs().maxCoeff();
    for (auto& [id, lm] : landmarks) {
      if (!lm.used) continue;
      const Eigen::Vector3d df = lm.Lff_inv * (lm.gf - lm.Lxf.transpose() * dx);
      lm.p += df;
      // Relative to the point's distance: depth converges slowly for distant
      // points, whose bearings (all the measurements see) are already fixed.
      step = std::max(step, df.norm() / std::max((lm.p - x[0].p_GI).norm(), 1.0));
    }
    res.steps.push_back(step);
    if (!std::isfinite(step)) break;
    for (int k = 0; k < N; ++k) {
      x[k].R_GI = x[k].R_GI * Sophus::SO3d::exp(dx.segment<3>(9 * k));
      x[k].p_GI += dx.segment<3>(9 * k + 3);
      x[k].v_GI += dx.segment<3>(9 * k + 6);
    }
    bg += dx.segment<3>(kB);
    ba += dx.segment<3>(kB + 3);
    if (!gated && step < config.gate_convergence) {
      gate_now = true;  // phase 1 done: close enough to gate; then refine again
      continue;
    }
    if (gated && step < config.convergence) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    res.reason = "diverged";
    return res;
  }

  // Hand-off: the last clone, the biases and the landmarks, with their joint
  // covariance from the Schur complement: Sxx = S^-1, Sfx = -Lff^-1 Lxf^T Sxx,
  // Sfg = Lff^-1 delta_fg + Lff^-1 Lxf^T Sxx Lxg Lgg^-1.
  res.clones = x;
  res.gyro_bias = bg;
  res.acc_bias = ba;
  ImuInitState& s = res.state;
  s.t = frames.back().t;
  s.nav = x.back();
  s.gyro_bias = bg;
  s.acc_bias = ba;
  std::vector<int> idx;  // the hand-off's IMU error, in imu_err order
  for (int i = 0; i < 9; ++i) idx.push_back(9 * (N - 1) + i);
  for (int i = 0; i < 6; ++i) idx.push_back(kB + i);
  Eigen::MatrixXd Sel = Eigen::MatrixXd::Zero(imu_err::kDim, n);
  for (int a = 0; a < imu_err::kDim; ++a) Sel(a, idx[a]) = 1.0;
  s.P = Sel * cov * Sel.transpose();
  if (Eigen::LLT<Matrix15d>(s.P).info() != Eigen::Success) {
    res.reason = "covariance";
    return res;
  }
  std::vector<const Landmark*> used;
  for (const auto& [id, lm] : landmarks) {
    if (lm.used) {
      used.push_back(&lm);
      res.landmarks.push_back({id, lm.p});
    }
  }
  const int M = static_cast<int>(used.size());
  std::vector<Eigen::MatrixXd> A(M);  // Lff^-1 Lxf^T: 3 x n
  for (int j = 0; j < M; ++j) A[j] = used[j]->Lff_inv * used[j]->Lxf.transpose();
  const int d = imu_err::kDim + 3 * M;
  res.P_joint = Eigen::MatrixXd::Zero(d, d);
  res.P_joint.topLeftCorner(imu_err::kDim, imu_err::kDim) = s.P;
  std::vector<Eigen::MatrixXd> ACov(M);  // A_j Sxx: 3 x n, once per landmark
  for (int j = 0; j < M; ++j) ACov[j] = A[j] * cov;
  for (int j = 0; j < M; ++j) {
    const Eigen::MatrixXd Sfx = -ACov[j];  // 3 x n
    res.P_joint.block(imu_err::kDim + 3 * j, 0, 3, imu_err::kDim) = Sfx * Sel.transpose();
    res.P_joint.block(0, imu_err::kDim + 3 * j, imu_err::kDim, 3) = (Sfx * Sel.transpose()).transpose();
    for (int l = j; l < M; ++l) {
      Eigen::Matrix3d Sfg = ACov[j] * A[l].transpose();
      if (l == j) Sfg += used[j]->Lff_inv;
      res.P_joint.block<3, 3>(imu_err::kDim + 3 * j, imu_err::kDim + 3 * l) = Sfg;
      res.P_joint.block<3, 3>(imu_err::kDim + 3 * l, imu_err::kDim + 3 * j) = Sfg.transpose();
    }
  }
  res.ok = true;
  return res;
}

}  // namespace msckf
