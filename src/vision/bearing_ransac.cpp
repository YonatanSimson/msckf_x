#include "msckf/vision/bearing_ransac.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace msckf::vision {

namespace {

// Weighted linear fit of E with f_b^T E f_a = 0 (8 or more correspondences),
// projected onto the essential manifold (two equal singular values, one zero).
// Row i is scaled by w[i]; with w = 1 / |[E_prev f_a]_xy-normal| the algebraic
// residual approximates the angle to the epipolar plane.
Eigen::Matrix3d fitEssential(const std::vector<Eigen::Vector3d>& f_a,
                             const std::vector<Eigen::Vector3d>& f_b, const std::vector<int>& idx,
                             const std::vector<double>* w = nullptr) {
  Eigen::MatrixXd A(idx.size(), 9);
  for (size_t r = 0; r < idx.size(); ++r) {
    const Eigen::Vector3d& a = f_a[idx[r]];
    const Eigen::Vector3d& b = f_b[idx[r]];
    const double s = w ? (*w)[r] : 1.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) A(r, 3 * i + j) = s * b[i] * a[j];
    }
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd e = svd.matrixV().col(8);
  Eigen::Matrix3d E;
  E << e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8];
  Eigen::JacobiSVD<Eigen::Matrix3d> s(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const double sigma = 0.5 * (s.singularValues()[0] + s.singularValues()[1]);
  return s.matrixU() * Eigen::Vector3d(sigma, sigma, 0.0).asDiagonal() * s.matrixV().transpose();
}

struct Score {
  double cost = std::numeric_limits<double>::infinity();  // MSAC: sum min(r^2, t^2); RANSAC: -inliers
  int inliers = 0;
};

}  // namespace

double epipolarAngle(const Eigen::Matrix3d& E, const Eigen::Vector3d& f_a, const Eigen::Vector3d& f_b) {
  const Eigen::Vector3d n_b = E * f_a;               // plane normal in frame b
  const Eigen::Vector3d n_a = E.transpose() * f_b;   // plane normal in frame a
  const double s_b = n_b.norm() > 0.0 ? std::abs(f_b.dot(n_b)) / (n_b.norm() * f_b.norm()) : 1.0;
  const double s_a = n_a.norm() > 0.0 ? std::abs(f_a.dot(n_a)) / (n_a.norm() * f_a.norm()) : 1.0;
  return std::asin(std::min(1.0, std::max(s_a, s_b)));
}

BearingRansacResult bearingRansac(const std::vector<Eigen::Vector3d>& f_a,
                                  const std::vector<Eigen::Vector3d>& f_b,
                                  const BearingRansacConfig& config) {
  BearingRansacResult res;
  const int n = static_cast<int>(f_a.size());
  res.inlier.assign(n, false);
  if (n < 8 || f_b.size() != f_a.size()) return res;
  const bool msac = config.method == RobustMethod::kLoMsac;
  const double t = config.threshold_rad;

  auto score = [&](const Eigen::Matrix3d& E, double thresh) {
    Score s;
    s.cost = 0.0;
    for (int i = 0; i < n; ++i) {
      const double r = epipolarAngle(E, f_a[i], f_b[i]);
      if (r < thresh) ++s.inliers;
      s.cost += msac ? std::min(r * r, thresh * thresh) : 0.0;
    }
    if (!msac) s.cost = -s.inliers;
    return s;
  };
  auto inliersOf = [&](const Eigen::Matrix3d& E, double thresh) {
    std::vector<int> idx;
    for (int i = 0; i < n; ++i) {
      if (epipolarAngle(E, f_a[i], f_b[i]) < thresh) idx.push_back(i);
    }
    return idx;
  };
  // Local optimization (LO-RANSAC, Lebeda et al. 2012, iterated least
  // squares): refit to the inliers of a shrinking threshold, reweighting each
  // row so the algebraic residual approximates the angular one.
  auto localOptimize = [&](Eigen::Matrix3d E, Score s) {
    constexpr int kSteps = 4;
    constexpr double kMult = 3.0;
    for (int k = 0; k <= kSteps; ++k) {
      const double thresh = t * (kMult - (kMult - 1.0) * k / kSteps);
      const std::vector<int> idx = inliersOf(E, thresh);
      if (idx.size() < 8) break;
      std::vector<double> w(idx.size());
      for (size_t r = 0; r < idx.size(); ++r) {
        const Eigen::Vector3d n_b = E * f_a[idx[r]];
        w[r] = 1.0 / std::max(n_b.norm(), 1e-12);
      }
      const Eigen::Matrix3d E_new = fitEssential(f_a, f_b, idx, &w);
      const Score s_new = score(E_new, t);
      if (s_new.cost < s.cost) {
        E = E_new;
        s = s_new;
      }
    }
    return std::make_pair(E, s);
  };

  std::mt19937_64 rng(config.seed);
  std::vector<int> all(n);
  std::iota(all.begin(), all.end(), 0);
  Score best;
  Eigen::Matrix3d best_E = Eigen::Matrix3d::Zero();
  int needed = config.max_iterations;
  std::vector<int> sample(8);
  for (res.iterations = 0; res.iterations < needed; ++res.iterations) {
    for (int k = 0; k < 8; ++k) {  // 8 distinct indices, partial Fisher-Yates
      std::uniform_int_distribution<int> pick(k, n - 1);
      std::swap(all[k], all[pick(rng)]);
      sample[k] = all[k];
    }
    Eigen::Matrix3d E = fitEssential(f_a, f_b, sample);
    Score s = score(E, t);
    if (s.cost >= best.cost) continue;
    if (msac) std::tie(E, s) = localOptimize(E, s);
    best = s;
    best_E = E;
    const double p8 = std::pow(static_cast<double>(s.inliers) / n, 8);
    if (p8 >= 1.0 - 1e-12) {
      needed = res.iterations + 1;
    } else if (p8 > 0.0) {
      // Clamp in floating point: for small inlier ratios k exceeds INT_MAX.
      const double k = std::log(1.0 - config.confidence) / std::log1p(-p8);
      needed = static_cast<int>(std::min<double>(config.max_iterations, std::ceil(k)));
    }
  }

  if (!msac) {
    // Plain RANSAC: one unweighted refit to the inliers, kept if it is better.
    const std::vector<int> idx = inliersOf(best_E, t);
    if (idx.size() >= 8) {
      const Eigen::Matrix3d E = fitEssential(f_a, f_b, idx);
      const Score s = score(E, t);
      if (s.cost < best.cost) {
        best_E = E;
        best = s;
      }
    }
  }
  res.E = best_E;
  for (int i = 0; i < n; ++i) res.inlier[i] = epipolarAngle(best_E, f_a[i], f_b[i]) < t;
  res.num_inliers = best.inliers;
  res.ok = true;
  return res;
}

}  // namespace msckf::vision
