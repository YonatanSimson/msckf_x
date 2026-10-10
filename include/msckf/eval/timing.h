#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace msckf::eval {

// Collects per-module latencies (seconds). Not thread-safe: one recorder per
// thread, merged at the end, keeps timing off the critical path.
class LatencyRecorder {
 public:
  void add(const std::string& module, double seconds) { samples_[module].push_back(seconds); }
  void merge(const LatencyRecorder& other);

  struct Summary {
    size_t count = 0;
    double mean = 0.0, median = 0.0, p99 = 0.0, max = 0.0;  // seconds
  };
  std::map<std::string, Summary> summarize() const;

 private:
  std::map<std::string, std::vector<double>> samples_;
};

// Records the lifetime of the scope under a module name.
class ScopedTimer {
 public:
  ScopedTimer(LatencyRecorder& recorder, std::string module)
      : recorder_(recorder), module_(std::move(module)), start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    recorder_.add(module_, std::chrono::duration<double>(end - start_).count());
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  LatencyRecorder& recorder_;
  std::string module_;
  std::chrono::steady_clock::time_point start_;
};

// Process CPU load since construction or the last reset(), in cores: process
// CPU time (user + system, all threads) divided by wall time. 1.0 = one full core.
class CpuLoadMeter {
 public:
  CpuLoadMeter() { reset(); }
  void reset();
  double load() const;

 private:
  double cpu_start_ = 0.0;
  std::chrono::steady_clock::time_point wall_start_;
};

// Percentile with linear interpolation between order statistics (q in [0, 1]).
double percentile(std::vector<double> values, double q);

}  // namespace msckf::eval
