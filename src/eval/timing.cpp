#include "msckf/eval/timing.h"

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace msckf::eval {

namespace {
double processCpuSeconds() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  auto seconds = [](const timeval& tv) { return tv.tv_sec + 1e-6 * tv.tv_usec; };
  return seconds(usage.ru_utime) + seconds(usage.ru_stime);
}
}  // namespace

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double pos = std::clamp(q, 0.0, 1.0) * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (pos - lo) * (values[hi] - values[lo]);
}

void LatencyRecorder::merge(const LatencyRecorder& other) {
  for (const auto& [module, v] : other.samples_) {
    auto& dst = samples_[module];
    dst.insert(dst.end(), v.begin(), v.end());
  }
}

std::map<std::string, LatencyRecorder::Summary> LatencyRecorder::summarize() const {
  std::map<std::string, Summary> out;
  for (const auto& [module, v] : samples_) {
    Summary s;
    s.count = v.size();
    if (!v.empty()) {
      s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
      s.median = percentile(v, 0.5);
      s.p99 = percentile(v, 0.99);
      s.max = *std::max_element(v.begin(), v.end());
    }
    out[module] = s;
  }
  return out;
}

void CpuLoadMeter::reset() {
  cpu_start_ = processCpuSeconds();
  wall_start_ = std::chrono::steady_clock::now();
}

double CpuLoadMeter::load() const {
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start_).count();
  return wall > 0.0 ? (processCpuSeconds() - cpu_start_) / wall : 0.0;
}

}  // namespace msckf::eval
