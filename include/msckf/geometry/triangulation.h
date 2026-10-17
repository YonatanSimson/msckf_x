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

// Point from camera poses T_GC and unit bearings in each camera (plan, Stage 3
// "Feature update"): least squares on the perpendicular distances to the rays,
// then Gauss-Newton on the tangent-plane bearing residuals over the inverse
// depth anchored in the first camera (Mourikis & Roumeliotis, ICRA 2007), with
// the anchor's z axis along its measured bearing. parallax_deg is the largest
// angle between two rays. False for a point behind a camera, a non-positive
// inverse depth or rays too close to parallel.
bool triangulate(const std::vector<Sophus::SE3d>& T_GC, const std::vector<Eigen::Vector3d>& b, Eigen::Vector3d* X,
                 double* parallax_deg);

}  // namespace msckf
