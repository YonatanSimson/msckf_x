#include "msckf/imu/propagation.h"

#include "msckf/core/constants.h"

namespace msckf {

namespace {

// State derivative in the coordinates RK4 integrates: (q, v, p) with q a Hamilton
// quaternion stored as Eigen coefficients (x, y, z, w).
struct Derivative {
  Eigen::Vector4d dq;
  Eigen::Vector3d dv;
  Eigen::Vector3d dp;
};

struct RawState {
  Eigen::Vector4d q;  // (x, y, z, w), not necessarily unit inside a step
  Eigen::Vector3d v;
  Eigen::Vector3d p;
};

Derivative derivative(const RawState& s, const ImuReading& r) {
  const Eigen::Quaterniond q(s.q[3], s.q[0], s.q[1], s.q[2]);
  const Eigen::Quaterniond w(0.0, r.omega.x(), r.omega.y(), r.omega.z());
  Derivative d;
  d.dq = 0.5 * (q * w).coeffs();
  // The rotation of a non-unit q scales by |q|^2; normalize for the velocity term.
  d.dv = q.normalized() * r.acc + kGravity;
  d.dp = s.v;
  return d;
}

RawState advance(const RawState& s, const Derivative& d, double h) {
  return {s.q + h * d.dq, s.v + h * d.dv, s.p + h * d.dp};
}

}  // namespace

NavState rk4Step(const NavState& x, const ImuReading& r0, const ImuReading& r_mid,
                 const ImuReading& r1, double dt) {
  const RawState s0{x.R_GI.unit_quaternion().coeffs(), x.v_GI, x.p_GI};
  const Derivative k1 = derivative(s0, r0);
  const Derivative k2 = derivative(advance(s0, k1, 0.5 * dt), r_mid);
  const Derivative k3 = derivative(advance(s0, k2, 0.5 * dt), r_mid);
  const Derivative k4 = derivative(advance(s0, k3, dt), r1);

  const double h6 = dt / 6.0;
  const Eigen::Vector4d q = s0.q + h6 * (k1.dq + 2.0 * k2.dq + 2.0 * k3.dq + k4.dq);

  NavState out;
  out.R_GI = Sophus::SO3d(Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized());
  out.v_GI = s0.v + h6 * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
  out.p_GI = s0.p + h6 * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
  return out;
}

ImuReading interpolate(const ImuReading& a, const ImuReading& b, double alpha) {
  return {(1.0 - alpha) * a.omega + alpha * b.omega, (1.0 - alpha) * a.acc + alpha * b.acc};
}

}  // namespace msckf
