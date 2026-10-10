#include "msckf/eval/metrics.h"

#include <algorithm>
#include <cmath>

namespace msckf::eval {

namespace {
constexpr double kRadToDeg = 180.0 / M_PI;
}

ErrorStats computeStats(std::vector<double> values) {
  ErrorStats s;
  s.count = values.size();
  if (values.empty()) return s;
  double sum = 0.0, sum_sq = 0.0;
  for (double v : values) {
    sum += v;
    sum_sq += v * v;
    s.max = std::max(s.max, v);
  }
  const double n = static_cast<double>(values.size());
  s.mean = sum / n;
  s.rmse = std::sqrt(sum_sq / n);
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  s.median = values[mid];
  if (values.size() % 2 == 0) {
    s.median = 0.5 * (s.median + *std::max_element(values.begin(), values.begin() + mid));
  }
  return s;
}

AteResult computeAte(const Association& assoc, AlignmentType type, double align_window) {
  AteResult out;
  out.type = type;
  std::vector<Eigen::Vector3d> est, gt;
  for (size_t i = 0; i < assoc.size(); ++i) {
    if (align_window > 0.0 && assoc.t[i] - assoc.t.front() > align_window) break;
    est.push_back(assoc.est[i].translation());
    gt.push_back(assoc.gt[i].translation());
  }
  out.alignment = align(est, gt, type);

  std::vector<double> pos_err, rot_err;
  pos_err.reserve(assoc.size());
  rot_err.reserve(assoc.size());
  for (size_t i = 0; i < assoc.size(); ++i) {
    const Sophus::SE3d T = out.alignment.apply(assoc.est[i]);
    pos_err.push_back((T.translation() - assoc.gt[i].translation()).norm());
    rot_err.push_back((assoc.gt[i].so3().inverse() * T.so3()).log().norm() * kRadToDeg);
  }
  out.position = computeStats(std::move(pos_err));
  out.rotation = computeStats(std::move(rot_err));
  return out;
}

double pathLength(const Association& assoc) {
  double d = 0.0;
  for (size_t i = 1; i < assoc.size(); ++i) {
    d += (assoc.gt[i].translation() - assoc.gt[i - 1].translation()).norm();
  }
  return d;
}

std::vector<double> defaultSegmentLengths(double path_length) {
  std::vector<double> out;
  for (double frac : {0.1, 0.2, 0.3, 0.4, 0.5}) {
    const double raw = frac * path_length;
    // Whole meters from 10 m up, 0.5 m steps below.
    const double nice = raw >= 10.0 ? std::round(raw) : std::max(0.5, std::round(raw * 2.0) / 2.0);
    if (out.empty() || nice > out.back()) out.push_back(nice);
  }
  return out;
}

std::vector<RelativeErrorResult> computeRelativeError(const Association& assoc,
                                                      const std::vector<double>& lengths) {
  std::vector<double> dist(assoc.size(), 0.0);
  for (size_t i = 1; i < assoc.size(); ++i) {
    dist[i] = dist[i - 1] + (assoc.gt[i].translation() - assoc.gt[i - 1].translation()).norm();
  }

  std::vector<RelativeErrorResult> out;
  for (double L : lengths) {
    RelativeErrorResult r;
    r.length = L;
    std::vector<double> trans, rot;
    for (size_t i = 0; i < assoc.size(); ++i) {
      const auto it = std::lower_bound(dist.begin() + i, dist.end(), dist[i] + L);
      if (it == dist.end()) break;  // later starts can't reach L either
      const size_t j = static_cast<size_t>(it - dist.begin());
      const Sophus::SE3d dT_gt = assoc.gt[i].inverse() * assoc.gt[j];
      const Sophus::SE3d dT_est = assoc.est[i].inverse() * assoc.est[j];
      const Sophus::SE3d E = dT_gt.inverse() * dT_est;
      // Normalize by the segment's actual path length, which exceeds L by up to
      // one sample spacing; normalizing by L would bias errors upward.
      const double d = dist[j] - dist[i];
      trans.push_back(100.0 * E.translation().norm() / d);
      rot.push_back(E.so3().log().norm() * kRadToDeg / d);
    }
    r.translation_pct = computeStats(std::move(trans));
    r.rotation_deg_per_m = computeStats(std::move(rot));
    out.push_back(r);
  }
  return out;
}

}  // namespace msckf::eval
