#pragma once

#include "msckf/camera/camera_model.h"
#include "msckf/filter/filter_state.h"

#include <Eigen/Core>

#include <cstdint>
#include <functional>
#include <vector>

namespace msckf::filter {

// One feature's observations, each in the image of a clone (matched by the
// camera timestamp).
struct FeatureTrack {
  int64_t id = 0;
  std::vector<double> t_cam;
  std::vector<Eigen::Vector2d> uv;
  std::vector<Eigen::Vector3d> bearing;  // unit, camera frame (camera model's unproject)
};

struct UpdateOptions {
  double pixel_noise = 1.0;        // px, per axis
  double chi2_probability = 0.95;  // gate on the projected residual
  double min_parallax_deg = 1.0;   // triangulation: largest angle between two rays
  int min_observations = 3;
  // Simulation diagnosis only: the true landmark position. Used features then
  // record their depth error (UpdateStats::depth_log_ratio); with
  // oracle_jacobians the Jacobians are evaluated at the true landmark (the
  // residual still uses the triangulated one); with oracle_selection the
  // parallax gate uses the rays to the true landmark (no selection on noise).
  std::function<bool(int64_t id, Eigen::Vector3d* p_G)> landmark_oracle;
  bool oracle_jacobians = false;
  bool oracle_selection = false;
};

struct UpdateStats {
  int features = 0;           // candidates
  int used = 0;               // passed every check
  int rejected_triangulation = 0;
  int rejected_chi2 = 0;
  int rows = 0;               // measurement rows after nullspace projection
  double nis = 0.0;           // sum of the used features' chi-square statistics (E = rows)
  // With a landmark oracle: per used feature, log(estimated / true depth) in
  // the first observing camera.
  std::vector<double> depth_log_ratio;
};

// Measurement model of one feature in clone i's image:
//   p_I = R_GI^T (p_f - p_GI),  p_C = R_IC^T (p_I - p_IC),  uv = pi(p_C).
// Blocks of d(p_C) (right perturbations): clone dtheta [p_I]x, clone dp
// -R_GI^T, feature R_GI^T, each times R_IC^T; extrinsic dtheta_IC [p_C]x,
// dp_IC -R_IC^T. The time offset enters through the clones' cloning Jacobian
// only. With FEJ the clone blocks use the clones' first estimates; the
// feature is one triangulated estimate (as in MSCKF 2.0, Li & Mourikis
// 2013: the yaw and translation nullspace holds for any feature point); the
// residual always uses the current estimates.
//
// Linearized feature residual: rows of H_x (dim() wide) and H_f (3 wide) and
// the pixel residual r = z - pi(...), two rows per observation. The residual
// uses p_f, the Jacobians p_f_lin when given (else p_f). False if a point
// leaves the camera model's valid region.
struct FeatureJacobian {
  Eigen::MatrixXd H_x;
  Eigen::MatrixXd H_f;
  Eigen::VectorXd r;
};
bool linearizeFeature(const FilterState& s, const CameraModel& camera, const FeatureTrack& track,
                      const std::vector<size_t>& clone_of_obs, const Eigen::Vector3d& p_f, FeatureJacobian* out,
                      const Eigen::Vector3d* p_f_lin = nullptr);

// MSCKF update with the given tracks (plan, Stage 3 "Feature update"): per
// feature, triangulate from the current clone estimates, linearize, project
// onto the left nullspace of H_f, chi-square gate; then stack, compress by QR
// when there are more rows than states, and apply a Joseph-form update.
// Observations at times without a clone are ignored.
UpdateStats featureUpdate(FilterState& s, const CameraModel& camera, const std::vector<FeatureTrack>& tracks,
                          const UpdateOptions& options);

// Chi-square quantile, cached per degrees of freedom.
double chi2Quantile(double probability, int dof);

}  // namespace msckf::filter
