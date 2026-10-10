#pragma once

#include <cmath>

namespace msckf::internal {

// Returns the end of the interval [0, x] on which derivative(x) > 0, searching up
// to cap: a coarse scan for the first sign change, then bisection. Used to find
// where a distortion polynomial stops being monotonic (and thus invertible).
template <typename F>
double monotonicLimit(F derivative, double cap, int coarse_steps = 256) {
  const double step = cap / coarse_steps;
  for (int i = 1; i <= coarse_steps; ++i) {
    const double hi = i * step;
    if (derivative(hi) > 0.0) continue;
    double lo = hi - step;
    double h = hi;
    for (int k = 0; k < 60; ++k) {
      const double mid = 0.5 * (lo + h);
      (derivative(mid) > 0.0 ? lo : h) = mid;
    }
    return lo;
  }
  return cap;
}

}  // namespace msckf::internal
