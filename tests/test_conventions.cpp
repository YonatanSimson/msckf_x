// Day-one convention tests: Hamilton quaternions, R_GI (body to global), right
// perturbation R = R_hat Exp(dtheta), and the mapping to the JPL algebra used in
// the MSCKF papers.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <random>

namespace {

constexpr double kTol = 1e-12;

// JPL quaternion stored as (x, y, z, w), the papers' layout.
using JplQuat = Eigen::Vector4d;

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// Trawny & Roumeliotis: C(q) = (2w^2 - 1) I - 2w [v x] + 2 v v^T.
Eigen::Matrix3d jplToRot(const JplQuat& q) {
  const Eigen::Vector3d v = q.head<3>();
  const double w = q.w();
  return (2.0 * w * w - 1.0) * Eigen::Matrix3d::Identity() - 2.0 * w * skew(v) +
         2.0 * v * v.transpose();
}

// JPL product (note the minus on the cross term), with C(q (x) p) = C(q) C(p).
JplQuat jplMul(const JplQuat& q, const JplQuat& p) {
  JplQuat out;
  out.head<3>() = q.w() * p.head<3>() + p.w() * q.head<3>() - q.head<3>().cross(p.head<3>());
  out.w() = q.w() * p.w() - q.head<3>().dot(p.head<3>());
  return out;
}

// JPL quaternion of a small rotation dtheta: [sin(|d|/2) n, cos(|d|/2)].
JplQuat jplFromRotationVector(const Eigen::Vector3d& dtheta) {
  const double angle = dtheta.norm();
  JplQuat q;
  if (angle < 1e-12) {
    q << 0.5 * dtheta, 1.0;
    return q.normalized();
  }
  q << std::sin(0.5 * angle) * dtheta / angle, std::cos(0.5 * angle);
  return q;
}

class ConventionsTest : public ::testing::Test {
 protected:
  Sophus::SO3d randomRotation() {
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::Quaterniond q(n(rng_), n(rng_), n(rng_), n(rng_));
    q.normalize();
    return Sophus::SO3d(q);
  }
  Eigen::Vector3d randomVector(double scale) {
    std::uniform_real_distribution<double> u(-scale, scale);
    return Eigen::Vector3d(u(rng_), u(rng_), u(rng_));
  }
  std::mt19937 rng_{42};
};

TEST_F(ConventionsTest, EigenQuaternionConstructorVsStorageOrder) {
  // The constructor takes (w, x, y, z) but coeffs() stores (x, y, z, w).
  const Eigen::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  EXPECT_EQ(q.w(), 1.0);
  EXPECT_EQ(q.x(), 2.0);
  EXPECT_EQ(q.coeffs()[0], 2.0);
  EXPECT_EQ(q.coeffs()[3], 1.0);
}

TEST_F(ConventionsTest, RoundTripQuaternionMatrixExpLog) {
  for (int i = 0; i < 1000; ++i) {
    const Sophus::SO3d R = randomRotation();
    const Eigen::Quaterniond q = R.unit_quaternion();

    const Sophus::SO3d from_matrix(q.toRotationMatrix());
    EXPECT_LT((from_matrix.matrix() - R.matrix()).norm(), kTol);

    const Sophus::SO3d from_exp = Sophus::SO3d::exp(R.log());
    EXPECT_LT((from_exp.matrix() - R.matrix()).norm(), kTol);

    // Quaternions match up to sign.
    const Eigen::Quaterniond q2 = from_exp.unit_quaternion();
    EXPECT_NEAR(std::abs(q.dot(q2)), 1.0, kTol);

    // Exp of a rotation vector equals Eigen's angle-axis.
    const Eigen::Vector3d w = randomVector(3.0);
    const Eigen::Matrix3d aa = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix();
    EXPECT_LT((Sophus::SO3d::exp(w).matrix() - aa).norm(), kTol);
    EXPECT_LT((Sophus::SO3d::hat(w) - skew(w)).norm(), kTol);
  }
}

TEST_F(ConventionsTest, RotationMapsBodyToGlobal) {
  // R_GI maps IMU-frame vectors to the global frame: a body x-axis yawed by 90
  // degrees points along global y.
  const Sophus::SO3d R_GI = Sophus::SO3d::rotZ(M_PI / 2.0);
  EXPECT_LT((R_GI * Eigen::Vector3d::UnitX() - Eigen::Vector3d::UnitY()).norm(), kTol);
}

TEST_F(ConventionsTest, JplQuaternionOfAttitudeIsHamiltonOfRGI) {
  // The JPL quaternion I_G q (C maps G -> I) has the same components as the
  // Hamilton quaternion of R_GI, and C = R_GI^T.
  for (int i = 0; i < 1000; ++i) {
    const Sophus::SO3d R_GI = randomRotation();
    const JplQuat q_jpl = R_GI.unit_quaternion().coeffs();  // (x, y, z, w)
    EXPECT_LT((jplToRot(q_jpl) - R_GI.matrix().transpose()).norm(), kTol);
  }
}

TEST_F(ConventionsTest, JplProductIsReversedHamiltonProduct) {
  for (int i = 0; i < 1000; ++i) {
    const Sophus::SO3d A = randomRotation(), B = randomRotation();
    const JplQuat qa = A.unit_quaternion().coeffs(), qb = B.unit_quaternion().coeffs();
    EXPECT_LT((jplToRot(jplMul(qa, qb)) - jplToRot(qa) * jplToRot(qb)).norm(), kTol);
    const Eigen::Vector4d hamilton_ba = (B.unit_quaternion() * A.unit_quaternion()).coeffs();
    EXPECT_LT((jplMul(qa, qb) - hamilton_ba).norm(), kTol);
  }
}

TEST_F(ConventionsTest, RightPerturbationMatchesJplLeftErrorQuaternion) {
  // R_GI = R_hat Exp(dtheta)  <=>  q_JPL = dq (x) q_hat_JPL, with the same dtheta.
  for (int i = 0; i < 1000; ++i) {
    const Sophus::SO3d R_hat = randomRotation();
    const Eigen::Vector3d dtheta = randomVector(0.5);

    const Eigen::Matrix3d R_true = (R_hat * Sophus::SO3d::exp(dtheta)).matrix();
    const JplQuat q_hat = R_hat.unit_quaternion().coeffs();
    const JplQuat q_true = jplMul(jplFromRotationVector(dtheta), q_hat);
    EXPECT_LT((jplToRot(q_true) - R_true.transpose()).norm(), kTol);
  }
}

}  // namespace
