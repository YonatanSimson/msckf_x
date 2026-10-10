#pragma once

#include "msckf/eval/alignment.h"
#include "msckf/eval/trajectory.h"

#include <vector>

namespace msckf::eval {

struct ErrorStats {
  double rmse = 0.0, mean = 0.0, median = 0.0, max = 0.0;
  size_t count = 0;
};

ErrorStats computeStats(std::vector<double> values);

struct AteResult {
  ErrorStats position;     // m
  ErrorStats rotation;     // deg
  Alignment alignment;
  AlignmentType type = AlignmentType::kPosYaw;
};

// Absolute trajectory error after aligning the estimate to ground truth.
// align_window > 0 fits the alignment on the first align_window seconds only
// (e.g. the Stage 1 Sim(3) scale fit on the first 10 s); errors are always
// computed over the whole association.
AteResult computeAte(const Association& assoc, AlignmentType type, double align_window = 0.0);

// Relative error over sub-trajectories of a given ground-truth path length,
// rpg_trajectory_evaluation style. For each start pose i, j is the first later
// pose with path length >= L; the error is E = dT_gt^-1 dT_est with
// dT = T_i^-1 T_j, which is independent of the global alignment. Errors are
// normalized by the segment's actual ground-truth path length d_ij >= L.
struct RelativeErrorResult {
  double length = 0.0;            // nominal segment length L, m
  ErrorStats translation_pct;     // 100 |t_E| / d_ij
  ErrorStats rotation_deg_per_m;  // |Log(R_E)| / d_ij, deg/m
};

std::vector<RelativeErrorResult> computeRelativeError(const Association& assoc,
                                                      const std::vector<double>& lengths);

// Ground-truth path length of the association.
double pathLength(const Association& assoc);

// rpg default segment lengths: 10, 20, 30, 40, 50 % of the path length,
// rounded to whole meters from 10 m up and to 0.5 m steps below.
std::vector<double> defaultSegmentLengths(double path_length);

}  // namespace msckf::eval
