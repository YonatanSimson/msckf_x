#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <vector>

namespace msckf {

using Matrix32d = Eigen::Matrix<double, 3, 2>;

// Orthonormal basis of the plane orthogonal to the unit vector b.
Matrix32d tangentBasis(const Eigen::Vector3d& b);

// Largest angle between two of the directions (any length), deg.
double maxRayAngleDeg(const std::vector<Eigen::Vector3d>& directions);

// A point in anchored inverse depth (Mourikis & Roumeliotis, ICRA 2007): in the
// fixed frame A, p_A = [alpha, beta, 1] / rho with x = (alpha, beta, rho).
// rho = 0 is a point at infinity in the direction R_GA [alpha, beta, 1]: it
// carries no depth but still constrains rotation.
struct AnchoredPoint {
  Sophus::SE3d T_GA;
  Eigen::Vector3d x = Eigen::Vector3d::Zero();

  // rho (p - c) for a position c: the point's direction from c, scaled so it
  // stays finite at infinity.
  Eigen::Vector3d scaledFrom(const Eigen::Vector3d& c) const;
  // The point in G; throws for rho <= 0.
  Eigen::Vector3d point() const;
  // Throws for a point behind (or on the z = 0 plane of) the anchor.
  static AnchoredPoint fromPoint(const Sophus::SE3d& T_GA, const Eigen::Vector3d& p_G);
};

// Point from camera poses T_GC and unit bearings in each camera (plan, Stage 3
// "Feature update"): Gauss-Newton on the tangent-plane bearing residuals over
// the inverse depth anchored in the first camera, with the anchor's z axis
// along its measured bearing; started from the midpoint of the rays, or at
// infinity when they do not meet in front (no baseline leaves rho at 0).
// parallax_deg is the largest angle between two rays. False for a non-finite
// solution, rho < 0 (rays that meet behind the cameras) or a ray pointing away
// from the point.
bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, AnchoredPoint* p,
                 double* parallax_deg);

}  // namespace msckf
