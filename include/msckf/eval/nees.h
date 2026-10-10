#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <vector>

namespace msckf::eval {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

// Regularized lower incomplete gamma P(a, x), and the chi-square quantile
// (inverse CDF) for k degrees of freedom, computed exactly by bisection on it.
double regularizedGammaP(double a, double x);
double chiSquareQuantile(double p, double dof);

// Two-sided 95% band for the average NEES over num_runs Monte Carlo runs of a
// dof-dimensional error: [chi2_0.025(N d) / N, chi2_0.975(N d) / N].
struct NeesBand {
  double lower = 0.0, upper = 0.0;
};
NeesBand neesBand95(int dof, int num_runs);

// Pose NEES for the right-perturbation convention used by the filter:
// R_true = R_hat Exp(dtheta), p_true = p_hat + dp, error e = [dtheta; dp], and
// cov the 6x6 covariance of e in that order.
struct PoseNees {
  double pose = 0.0;         // 6 dof
  double orientation = 0.0;  // 3 dof, from the orientation block
  double position = 0.0;     // 3 dof, from the position block
};
PoseNees poseNees(const Sophus::SE3d& T_est, const Matrix6d& cov, const Sophus::SE3d& T_true);

// Averages NEES across Monte Carlo runs that share a time grid (same number of
// epochs, e.g. one entry per camera frame).
class MonteCarloNees {
 public:
  void addRun(const std::vector<PoseNees>& run);

  int numRuns() const { return num_runs_; }
  size_t numEpochs() const { return sum_.size(); }

  // Average NEES per epoch.
  std::vector<PoseNees> average() const;

  struct Summary {
    double mean_pose = 0.0, mean_orientation = 0.0, mean_position = 0.0;  // over epochs
    NeesBand band6, band3;
    // Fraction of epochs whose average NEES lies inside the band.
    double inside_pose = 0.0, inside_orientation = 0.0, inside_position = 0.0;
  };
  Summary summarize() const;

 private:
  int num_runs_ = 0;
  std::vector<PoseNees> sum_;
};

}  // namespace msckf::eval
