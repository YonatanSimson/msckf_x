#include "msckf/eval/report.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace msckf::eval {

namespace fs = std::filesystem;

std::vector<DatasetEntry> readDatasetList(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dataset list " + path);
  const fs::path base = fs::path(path).parent_path();
  auto resolve = [&](const std::string& p) {
    const fs::path fp(p);
    return (fp.is_absolute() ? fp : base / fp).lexically_normal().string();
  };

  std::vector<DatasetEntry> out;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    std::istringstream ss(line);
    DatasetEntry e;
    if (!(ss >> e.name)) continue;  // blank or comment
    if (!(ss >> e.gt_path >> e.est_path)) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) +
                               ": expected 'name gt_path est_path [gt_format]'");
    }
    ss >> e.gt_format;
    e.gt_path = resolve(e.gt_path);
    e.est_path = resolve(e.est_path);
    out.push_back(e);
  }
  return out;
}

DatasetResult evaluateDataset(const DatasetEntry& entry, const EvalOptions& options) {
  DatasetResult r;
  r.name = entry.name;
  try {
    const Trajectory gt = readTrajectory(entry.gt_path, entry.gt_format);
    const Trajectory est = readTum(entry.est_path);
    r.num_estimates = est.size();
    const Association assoc = associate(est, gt, options.max_gap);
    r.num_matched = assoc.size();
    if (assoc.size() < 3) {
      r.error = "fewer than 3 estimate poses overlap the ground truth";
      return r;
    }
    r.duration = assoc.t.back() - assoc.t.front();
    r.path_length = pathLength(assoc);
    r.ate = computeAte(assoc, options.alignment);
    const std::vector<double> lengths = options.segment_lengths.empty()
                                            ? defaultSegmentLengths(r.path_length)
                                            : options.segment_lengths;
    r.relative = computeRelativeError(assoc, lengths);
  } catch (const std::exception& ex) {
    r.error = ex.what();
  }
  return r;
}

namespace {

// Mean over segment lengths of the per-length means; NaN-free when empty.
double meanOverLengths(const std::vector<RelativeErrorResult>& re, bool translation) {
  double sum = 0.0;
  int n = 0;
  for (const auto& r : re) {
    const ErrorStats& s = translation ? r.translation_pct : r.rotation_deg_per_m;
    if (s.count == 0) continue;
    sum += s.mean;
    ++n;
  }
  return n > 0 ? sum / n : 0.0;
}

}  // namespace

std::string formatReport(const Report& report) {
  std::ostringstream md;
  md << std::fixed;
  md << "# " << report.title << "\n\n";
  md << "Alignment: " << toString(report.options.alignment)
     << ". ATE is RMSE over all matched poses after alignment. Relative error (RE) uses "
        "sub-trajectories of fixed ground-truth length; translation in % of length, rotation "
        "in deg/m.\n\n";

  md << "## Summary\n\n";
  md << "| Dataset | Matched / estimates | Duration (s) | Length (m) | ATE RMSE (m) | ATE rot RMSE "
        "(deg) | RE trans mean (%) | RE rot mean (deg/m) |\n";
  md << "|---|---|---|---|---|---|---|---|\n";
  for (const DatasetResult& d : report.datasets) {
    if (!d.error.empty()) {
      md << "| " << d.name << " | error: " << d.error << " | | | | | | |\n";
      continue;
    }
    md << "| " << d.name << " | " << d.num_matched << " / " << d.num_estimates << " | "
       << std::setprecision(1) << d.duration << " | " << d.path_length << " | "
       << std::setprecision(4) << d.ate.position.rmse << " | " << std::setprecision(3)
       << d.ate.rotation.rmse << " | " << std::setprecision(3)
       << meanOverLengths(d.relative, true) << " | " << std::setprecision(4)
       << meanOverLengths(d.relative, false) << " |\n";
  }

  for (const DatasetResult& d : report.datasets) {
    if (!d.error.empty()) continue;
    md << "\n## " << d.name << "\n\n";
    md << "ATE position (m): RMSE " << std::setprecision(4) << d.ate.position.rmse << ", mean "
       << d.ate.position.mean << ", median " << d.ate.position.median << ", max "
       << d.ate.position.max << ". Rotation (deg): RMSE " << std::setprecision(3)
       << d.ate.rotation.rmse << ", max " << d.ate.rotation.max << ".\n\n";
    if (d.ate.type == AlignmentType::kSim3) {
      md << "Sim(3) scale: " << std::setprecision(5) << d.ate.alignment.scale << "\n\n";
    }
    md << "| Segment (m) | Pairs | Trans mean (%) | Trans median (%) | Rot mean (deg/m) | Rot "
          "median (deg/m) |\n";
    md << "|---|---|---|---|---|---|\n";
    for (const RelativeErrorResult& r : d.relative) {
      md << "| " << std::setprecision(1) << r.length << " | " << r.translation_pct.count << " | "
         << std::setprecision(3) << r.translation_pct.mean << " | " << r.translation_pct.median
         << " | " << std::setprecision(4) << r.rotation_deg_per_m.mean << " | "
         << r.rotation_deg_per_m.median << " |\n";
    }
  }

  if (report.nees) {
    const auto& s = *report.nees;
    md << "\n## Consistency (NEES, " << report.nees_runs << " Monte Carlo runs)\n\n";
    md << "| Block | Dof | Mean ANEES | 95% band | Epochs inside band |\n|---|---|---|---|---|\n";
    auto row = [&](const char* name, int dof, double mean, const NeesBand& b, double inside) {
      md << "| " << name << " | " << dof << " | " << std::setprecision(3) << mean << " | ["
         << b.lower << ", " << b.upper << "] | " << std::setprecision(1) << 100.0 * inside
         << "% |\n";
    };
    row("Pose", 6, s.mean_pose, s.band6, s.inside_pose);
    row("Orientation", 3, s.mean_orientation, s.band3, s.inside_orientation);
    row("Position", 3, s.mean_position, s.band3, s.inside_position);
  }

  if (!report.timing.empty()) {
    md << "\n## Timing\n\n| Module | Calls | Median (ms) | p99 (ms) | Max (ms) |\n";
    md << "|---|---|---|---|---|\n";
    for (const auto& [module, s] : report.timing) {
      md << "| " << module << " | " << s.count << " | " << std::setprecision(3)
         << 1e3 * s.median << " | " << 1e3 * s.p99 << " | " << 1e3 * s.max << " |\n";
    }
  }
  return md.str();
}

void writeReport(const std::string& dir, const Report& report) {
  fs::create_directories(dir);
  {
    std::ofstream md(fs::path(dir) / "report.md");
    if (!md) throw std::runtime_error("cannot write report.md in " + dir);
    md << formatReport(report);
  }
  std::ofstream csv(fs::path(dir) / "metrics.csv");
  if (!csv) throw std::runtime_error("cannot write metrics.csv in " + dir);
  csv << "dataset,error,matched,estimates,duration_s,length_m,ate_rmse_m,ate_mean_m,"
         "ate_median_m,ate_max_m,ate_rot_rmse_deg,re_trans_mean_pct,re_rot_mean_deg_per_m\n";
  csv << std::setprecision(9);
  for (const DatasetResult& d : report.datasets) {
    std::string err = d.error;
    for (char& c : err) {
      if (c == ',' || c == '\n') c = ';';
    }
    csv << d.name << ',' << err << ',' << d.num_matched << ',' << d.num_estimates << ','
        << d.duration << ',' << d.path_length << ',' << d.ate.position.rmse << ','
        << d.ate.position.mean << ',' << d.ate.position.median << ',' << d.ate.position.max
        << ',' << d.ate.rotation.rmse << ',' << meanOverLengths(d.relative, true) << ','
        << meanOverLengths(d.relative, false) << '\n';
  }
}

}  // namespace msckf::eval
