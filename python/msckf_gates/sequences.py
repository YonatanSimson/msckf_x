"""Real-data sequences shared by the gates that run on them (Stage 1b
initialization, Stage 3 MSCKF): where each one's camera, IMU noise and ground
truth are, which front end tracks it, and the cached tracks. The lists cover
what is on disk (EuRoC MH and Vicon rooms, TUM-VI room1-2, the UZH-FPV
Snapdragon sequences with ground truth, the LaMAria shorts); nothing here
downloads data.

Tracks are tracker_eval's tracks.csv, cached in the Stage 1b gate's output
(gate_out/stage1b_dynamic_init/tracks_<name>), so a benchmark run tracks each
sequence once.
"""

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path

from msckf_gates.common import FRONT_END_EUROC, FRONT_END_TUMVI, gate_out, run, tool

EUROC_MH: list[str] = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
EUROC_VICON: list[str] = ["V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium",
                          "V2_03_difficult"]  # fmt: skip
EUROC: list[str] = EUROC_MH + EUROC_VICON  # Stage 3; Stage 1b gates on the Machine Hall only
# TUM-VI sequences the user chose (2026-10-20): the rooms have mocap truth over
# the whole trajectory; corridor and slides only in the mocap room at the start
# and end, so their ATE is end-to-end drift. Sequences not on disk are skipped.
TUMVI: list[str] = (
    [f"room{i}" for i in range(1, 7)] + [f"corridor{i}" for i in range(1, 6)] + [f"slides{i}" for i in range(1, 4)]
)
# UZH-FPV Snapdragon sequences with public ground truth (python/uzh_fpv_to_euroc.py).
UZH_FPV: list[str] = [
    "indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7", "indoor_forward_9",
    "indoor_forward_10", "indoor_45_2", "indoor_45_4", "indoor_45_9", "indoor_45_12", "indoor_45_13",
    "indoor_45_14", "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5", "outdoor_45_1",
]  # fmt: skip
# One LaMAria sequence per difficulty, from the MPS-ground-truth set (R_01-R_10;
# the control-point sequences' pGT has keyframes up to 1 s apart, too sparse
# for velocity truth). All ten cost ~20 min of tracking per run.
LAMARIA: list[str] = ["R_01_easy", "R_02_easy", "R_03_easy"]
# Real-data noise inflation, tuned by python/tune_init_noise.py on EuRoC
# V1_01-03 and TUM-VI room2 (no gate uses them): IMU noise densities x4 and
# 2 px bearings bring V1_02, V1_03 and room2 into their NEES bands (V1_01, slow
# motion, stays ~3x overconfident: open). Simulation is not inflated.
REAL_NOISE: list[str] = ["--imu-noise-scale", "4", "--pixel-sigma", "2"]


@dataclass(frozen=True)
class Sequence:
    name: str
    cam: list  # --euroc DIR --cam cam0 [--camchain YAML]
    front_end: list  # tracker_eval options
    gt: Path  # EuRoC-format ground truth
    bias: str  # "gt": EuRoC's ground-truth biases as prior calibration, else "zero"
    imu_yaml: Path  # IMU noise densities


def euroc(root: Path, names: list[str] = EUROC) -> list[Sequence]:
    return [
        Sequence(f"euroc_{s}", ["--euroc", root / s, "--cam", "cam0"], FRONT_END_EUROC,
                 root / s / "mav0" / "state_groundtruth_estimate0" / "data.csv", "gt",
                 root / s / "mav0" / "imu0" / "sensor.yaml")  # fmt: skip
        for s in names
    ]


def tumvi(root: Path, names: list[str] = TUMVI) -> list[Sequence]:
    """TUM-VI 512 sequences with their dso/ calibration and the published Kalibr IMU noise."""
    imu = root / "kalibr_results" / "pinhole-equi-512" / "imu-imucalib.yaml"
    out = []
    for name in names:
        d = root / f"dataset-{name}_512_16"
        cam = ["--euroc", d, "--cam", "cam0", "--camchain", d / "dso" / "camchain.yaml"]
        out.append(Sequence(f"tumvi_{name}", cam, FRONT_END_TUMVI, d / "dso" / "gt_imu.csv", "zero", imu))
    return out


def tumvi_room1(root: Path) -> Sequence:
    return tumvi(root, ["room1"])[0]


def lamaria_fisheye(dirs: list[Path]) -> list[Sequence]:
    """The raw fisheye export (python/lamaria_fisheye_to_euroc.py) of LaMAria sequences."""
    return [
        Sequence(f"lamaria_fisheye_{d.name}", ["--euroc", d / "fisheye", "--cam", "cam0"], FRONT_END_EUROC,
                 d / "fisheye" / "gt_imu.csv", "zero", d / "fisheye" / "mav0" / "imu0" / "sensor.yaml")  # fmt: skip
        for d in dirs
    ]


def converted(prefix: str, dirs: list[Path]) -> list[Sequence]:
    """Sequences converted to EuRoC layout (KITTI raw, LaMAria, UZH-FPV): gt_imu.csv, zero bias."""
    return [
        Sequence(f"{prefix}_{d.name}", ["--euroc", d, "--cam", "cam0"], FRONT_END_EUROC, d / "gt_imu.csv", "zero",
                 d / "mav0" / "imu0" / "sensor.yaml")  # fmt: skip
        for d in dirs
    ]


def tracks_csv(seq: Sequence) -> Path:
    return gate_out("stage1b_dynamic_init") / f"tracks_{seq.name}" / "tracks.csv"


def submit_tracking(ex: Executor, sequences: list[Sequence]) -> list[Future]:
    """Tracks every sequence whose tracks are not cached yet."""
    jobs = []
    for seq in sequences:
        if not tracks_csv(seq).exists():
            out = tracks_csv(seq).parent
            cmd = [tool("tracker_eval"), "tracks", *seq.cam, *seq.front_end, "--out", out]
            jobs.append(ex.submit(run, cmd, out.parent / f"{out.name}.log"))
    return jobs
