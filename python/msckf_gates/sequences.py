"""Real-data sequences shared by the gates that run on them (Stage 1b
initialization, Stage 3 MSCKF): where each one's camera, IMU noise and ground
truth are, which front end tracks it, and the cached tracks.

Tracks are tracker_eval's tracks.csv, cached in the Stage 1b gate's output
(gate_out/stage1b_dynamic_init/tracks_<name>), so a benchmark run tracks each
sequence once.
"""

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path

from msckf_gates.common import FRONT_END_EUROC, FRONT_END_TUMVI, gate_out, run, tool

EUROC: list[str] = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
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


def tumvi_room1(root: Path) -> Sequence:
    room1 = root / "dataset-room1_512_16"
    cam = ["--euroc", room1, "--cam", "cam0", "--camchain", room1 / "dso" / "camchain.yaml"]
    imu = root / "kalibr_results" / "pinhole-equi-512" / "imu-imucalib.yaml"
    return Sequence("tumvi_room1", cam, FRONT_END_TUMVI, room1 / "dso" / "gt_imu.csv", "zero", imu)


def converted(prefix: str, dirs: list[Path]) -> list[Sequence]:
    """Sequences converted to EuRoC layout (KITTI raw, LaMAria): gt_imu.csv, zero bias."""
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
