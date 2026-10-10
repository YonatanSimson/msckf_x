#include "msckf/eval/nees.h"

#include <Eigen/Cholesky>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace msckf::eval {

double regularizedGammaP(double a, double x) {
  if (a <= 0.0) throw std::invalid_argument("regularizedGammaP: a must be > 0");
  if (x <= 0.0) return 0.0;
  const double log_prefactor = a * std::log(x) - x - std::lgamma(a);
  constexpr double kEps = 1e-15;
  constexpr int kMaxIter = 10000;

  if (x < a + 1.0) {
    // Series: P = x^a e^-x / Gamma(a) * sum_n x^n / (a (a+1) ... (a+n)).
    double term = 1.0 / a, sum = term;
    for (int n = 1; n < kMaxIter; ++n) {
      term *= x / (a + n);
      sum += term;
      if (std::abs(term) < std::abs(sum) * kEps) break;
    }
    return std::exp(log_prefactor) * sum;
  }

  // Continued fraction for Q = 1 - P, evaluated with the modified Lentz method:
  // Q = x^a e^-x / Gamma(a) * 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a- ...))).
  constexpr double kTiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / kTiny;
  double d = 1.0 / b;
  double f = d;
  for (int n = 1; n < kMaxIter; ++n) {
    const double an = -n * (n - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < kTiny) d = kTiny;
    c = b + an / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double delta = d * c;
    f *= delta;
    if (std::abs(delta - 1.0) < kEps) break;
  }
  return 1.0 - std::exp(log_prefactor) * f;
}

double chiSquareQuantile(double p, double dof) {
  if (p <= 0.0 || p >= 1.0 || dof <= 0.0) {
    throw std::invalid_argument("chiSquareQuantile: need 0 < p < 1, dof > 0");
  }
  // CDF(x) = P(dof/2, x/2), monotonic in x. Bracket, then bisect.
  double lo = 0.0, hi = std::max(1.0, dof);
  while (regularizedGammaP(0.5 * dof, 0.5 * hi) < p) hi *= 2.0;
  for (int i = 0; i < 200 && hi - lo > 1e-12 * hi; ++i) {
    const double mid = 0.5 * (lo + hi);
    (regularizedGammaP(0.5 * dof, 0.5 * mid) < p ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

NeesBand neesBand95(int dof, int num_runs) {
  const double n = num_runs, k = static_cast<double>(dof) * num_runs;
  return {chiSquareQuantile(0.025, k) / n, chiSquareQuantile(0.975, k) / n};
}

PoseNees poseNees(const Sophus::SE3d& T_est, const Matrix6d& cov, const Sophus::SE3d& T_true) {
  Eigen::Matrix<double, 6, 1> e;
  e.head<3>() = (T_est.so3().inverse() * T_true.so3()).log();
  e.tail<3>() = T_true.translation() - T_est.translation();

  auto nees = [](const auto& err, const auto& P) {
    const auto llt = P.llt();
    if (llt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
    return err.dot(llt.solve(err));
  };
  PoseNees out;
  out.pose = nees(e, cov);
  out.orientation = nees(Eigen::Vector3d(e.head<3>()), Eigen::Matrix3d(cov.topLeftCorner<3, 3>()));
  out.position = nees(Eigen::Vector3d(e.tail<3>()), Eigen::Matrix3d(cov.bottomRightCorner<3, 3>()));
  return out;
}

void MonteCarloNees::addRun(const std::vector<PoseNees>& run) {
  if (num_runs_ == 0) {
    sum_.assign(run.size(), PoseNees{});
  } else if (run.size() != sum_.size()) {
    throw std::invalid_argument("MonteCarloNees: runs must have the same number of epochs");
  }
  for (size_t k = 0; k < run.size(); ++k) {
    sum_[k].pose += run[k].pose;
    sum_[k].orientation += run[k].orientation;
    sum_[k].position += run[k].position;
  }
  ++num_runs_;
}

std::vector<PoseNees> MonteCarloNees::average() const {
  std::vector<PoseNees> out(sum_.size());
  for (size_t k = 0; k < sum_.size(); ++k) {
    out[k] = {sum_[k].pose / num_runs_, sum_[k].orientation / num_runs_,
              sum_[k].position / num_runs_};
  }
  return out;
}

MonteCarloNees::Summary MonteCarloNees::summarize() const {
  Summary s;
  if (num_runs_ == 0 || sum_.empty()) return s;
  s.band6 = neesBand95(6, num_runs_);
  s.band3 = neesBand95(3, num_runs_);
  auto inside = [](double v, const NeesBand& b) { return v >= b.lower && v <= b.upper ? 1.0 : 0.0; };
  for (const PoseNees& a : average()) {
    s.mean_pose += a.pose;
    s.mean_orientation += a.orientation;
    s.mean_position += a.position;
    s.inside_pose += inside(a.pose, s.band6);
    s.inside_orientation += inside(a.orientation, s.band3);
    s.inside_position += inside(a.position, s.band3);
  }
  const double n = static_cast<double>(sum_.size());
  s.mean_pose /= n;
  s.mean_orientation /= n;
  s.mean_position /= n;
  s.inside_pose /= n;
  s.inside_orientation /= n;
  s.inside_position /= n;
  return s;
}

}  // namespace msckf::eval
