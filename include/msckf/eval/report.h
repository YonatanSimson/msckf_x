#pragma once

#include "msckf/eval/metrics.h"
#include "msckf/eval/nees.h"
#include "msckf/eval/timing.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace msckf::eval {

// One line of a dataset list: "name gt_path est_path [gt_format]", where
// gt_format is "tum" (default) or "euroc" and the estimate is always TUM.
// Relative paths are resolved against the list file's directory.
struct DatasetEntry {
  std::string name;
  std::string gt_path;
  std::string est_path;
  std::string gt_format = "tum";
};

std::vector<DatasetEntry> readDatasetList(const std::string& path);

struct EvalOptions {
  AlignmentType alignment = AlignmentType::kPosYaw;
  std::vector<double> segment_lengths;  // m; empty = 10..50 % of each path length
  double max_gap = 0.05;                // s, for ground-truth interpolation
};

struct DatasetResult {
  std::string name;
  std::string error;  // non-empty if the dataset could not be evaluated
  size_t num_estimates = 0;
  size_t num_matched = 0;
  double duration = 0.0;     // s, of the matched poses
  double path_length = 0.0;  // m, ground truth
  AteResult ate;
  std::vector<RelativeErrorResult> relative;
};

DatasetResult evaluateDataset(const DatasetEntry& entry, const EvalOptions& options);

struct Report {
  std::string title = "Evaluation report";
  EvalOptions options;
  std::vector<DatasetResult> datasets;
  std::map<std::string, LatencyRecorder::Summary> timing;  // optional
  std::optional<MonteCarloNees::Summary> nees;              // optional, simulation only
  int nees_runs = 0;
};

// Writes report.md (human-readable, fixed layout) and metrics.csv (one row per
// dataset, fixed columns) into dir, creating it if needed.
void writeReport(const std::string& dir, const Report& report);

// The markdown body, also printed by the command-line tool.
std::string formatReport(const Report& report);

}  // namespace msckf::eval
