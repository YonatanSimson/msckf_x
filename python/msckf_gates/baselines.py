"""Published accuracy of the systems we compare against (plan, Stage 3 metrics:
"ATE within 20 % of OpenVINS mono"), quoted from papers only, never measured
by us: the clean-room rule allows reading papers and running binaries, not
reading their code.

Source: Peng, Chen, Wu, Huang, "sqrtVINS: Robust and Ultrafast Square-Root
Filter-based 3D Motion Tracking", T-RO 2025 (arXiv 2510.10346), Table VI:
average ATE in degrees / meters on EuRoC MAV, monocular (left camera), with
online calibration. OpenVINS: the default configuration in double precision
(11 clones, up to 50 SLAM and 40 MSCKF features). sqrtVINS: the same
configuration in double precision. The averaging and alignment are the
paper's (ov_eval); our msckf_eval reports RMSE after a position + yaw
alignment, so the ratios are indicative, not exact.
"""

from dataclasses import dataclass
from typing import Optional

from msckf_gates.common import fnum, gate_out, read_csv


@dataclass(frozen=True)
class PublishedAte:
    ate_m: float
    rot_deg: float


# Sequence name as in msckf_gates.sequences -> system -> ATE.
EUROC_MONO: dict[str, dict[str, PublishedAte]] = {
    "euroc_MH_01_easy": {"OpenVINS": PublishedAte(0.10, 1.74), "sqrtVINS": PublishedAte(0.11, 1.90)},
    "euroc_MH_02_easy": {"OpenVINS": PublishedAte(0.17, 0.91), "sqrtVINS": PublishedAte(0.14, 0.77)},
    "euroc_MH_03_medium": {"OpenVINS": PublishedAte(0.12, 1.14), "sqrtVINS": PublishedAte(0.12, 1.06)},
    "euroc_MH_04_difficult": {"OpenVINS": PublishedAte(0.25, 0.95), "sqrtVINS": PublishedAte(0.23, 1.02)},
    "euroc_MH_05_difficult": {"OpenVINS": PublishedAte(0.41, 1.03), "sqrtVINS": PublishedAte(0.35, 1.14)},
    "euroc_V1_01_easy": {"OpenVINS": PublishedAte(0.06, 0.70), "sqrtVINS": PublishedAte(0.06, 0.54)},
    "euroc_V1_02_medium": {"OpenVINS": PublishedAte(0.06, 1.67), "sqrtVINS": PublishedAte(0.06, 1.60)},
    "euroc_V1_03_difficult": {"OpenVINS": PublishedAte(0.07, 2.88), "sqrtVINS": PublishedAte(0.05, 2.94)},
    "euroc_V2_01_easy": {"OpenVINS": PublishedAte(0.10, 0.95), "sqrtVINS": PublishedAte(0.10, 1.09)},
    "euroc_V2_02_medium": {"OpenVINS": PublishedAte(0.06, 1.38), "sqrtVINS": PublishedAte(0.06, 1.41)},
    "euroc_V2_03_difficult": {"OpenVINS": PublishedAte(0.14, 1.28), "sqrtVINS": PublishedAte(0.12, 1.36)},
}

SYSTEMS = ("OpenVINS", "sqrtVINS")


def published(sequence: str) -> dict[str, PublishedAte]:
    """The published ATEs of a sequence, empty when none is known."""
    return EUROC_MONO.get(sequence, {})


def measured(system: str, sequence: str) -> Optional[float]:
    """ATE RMSE (m) of the system run as a binary on the sequence by the
    stage3_baselines benchmark (gate_out/stage3_baselines), None when absent."""
    metrics = gate_out("stage3_baselines") / system.lower() / sequence / "eval" / "metrics.csv"
    if not metrics.exists():
        return None
    row = read_csv(metrics)[0]
    return None if row["error"] else fnum(row["ate_rmse_m"])
