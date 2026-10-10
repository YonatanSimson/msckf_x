// Evaluates estimated trajectories against ground truth and writes a fixed
// report (report.md + metrics.csv).
//
//   msckf_eval --list datasets.txt [options]
//   msckf_eval --gt gt.txt --est est.txt [--name NAME] [--gt-format tum|euroc] [options]
//
// Options:
//   --align posyaw|se3|sim3|none   alignment for ATE (default posyaw)
//   --segments 5,10,20             relative-error segment lengths in m
//                                  (default 10..50 % of each path length)
//   --max-gap S                    max ground-truth gap for interpolation (default 0.05)
//   --out DIR                      output directory (default eval_out)
//   --title TEXT                   report title

#include "msckf/eval/report.h"

#include <iostream>
#include <sstream>
#include <string>

namespace {

void usage() {
  std::cerr << "usage: msckf_eval (--list FILE | --gt FILE --est FILE [--name NAME] "
               "[--gt-format tum|euroc]) [--align posyaw|se3|sim3|none] [--segments L1,L2,...] "
               "[--max-gap S] [--out DIR] [--title TEXT]\n";
}

std::vector<double> parseList(const std::string& s) {
  std::vector<double> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace msckf::eval;
  std::string list, gt, est, name = "dataset", gt_format = "tum", out = "eval_out";
  Report report;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--list") list = next();
      else if (arg == "--gt") gt = next();
      else if (arg == "--est") est = next();
      else if (arg == "--name") name = next();
      else if (arg == "--gt-format") gt_format = next();
      else if (arg == "--align") report.options.alignment = parseAlignment(next());
      else if (arg == "--segments") report.options.segment_lengths = parseList(next());
      else if (arg == "--max-gap") report.options.max_gap = std::stod(next());
      else if (arg == "--out") out = next();
      else if (arg == "--title") report.title = next();
      else if (arg == "-h" || arg == "--help") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("unknown argument " + arg);
      }
    }

    std::vector<DatasetEntry> entries;
    if (!list.empty()) {
      entries = readDatasetList(list);
    } else if (!gt.empty() && !est.empty()) {
      entries.push_back({name, gt, est, gt_format});
    } else {
      usage();
      return 2;
    }

    int failures = 0;
    for (const DatasetEntry& e : entries) {
      report.datasets.push_back(evaluateDataset(e, report.options));
      if (!report.datasets.back().error.empty()) ++failures;
    }
    writeReport(out, report);
    std::cout << formatReport(report) << "\nWrote " << out << "/report.md and " << out
              << "/metrics.csv\n";
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "msckf_eval: " << ex.what() << "\n";
    usage();
    return 2;
  }
}
