#!/usr/bin/env python3
"""Adds the EuRoC layout our tools read to UZH-FPV Snapdragon sequences
(the v3 text releases, extracted next to their zips):

  <seq>/mav0/cam0/data.csv, data   left camera (img/image_0_N.png, linked)
  <seq>/mav0/cam1/data.csv, data   right camera (img/image_1_N.png)
  <seq>/mav0/camN/sensor.yaml      pinhole-equidistant intrinsics, T_BS and
                                   timeshift_cam_imu from the published Kalibr
                                   camchain of the sequence's sensor group
  <seq>/mav0/imu0/data.csv         imu.txt (500 Hz) in EuRoC columns
  <seq>/mav0/imu0/sensor.yaml      noise densities from the group's imu.yaml
  <seq>/gt_imu.csv                 groundtruth.txt (Leica + IMU batch estimate,
                                   pose of the Snapdragon IMU, 20 Hz) as the
                                   pose-only TUM-VI gt_imu.csv layout

Sensor groups: indoor_forward, indoor_45, outdoor_forward, outdoor_45, each
with its own <group>_calib_snapdragon calibration. Timestamps are seconds in
the files and become nanoseconds. Prints the mean world-frame specific force
under the ground truth, (0, 0, +g) when the frames are right.

    python python/uzh_fpv_to_euroc.py [SEQ_DIR...]

SEQ_DIR defaults to every <MSCKF_DATA_LARGE>/uzh_fpv/*_snapdragon_with_gt.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from euroc_layout import (
    gravity_check_line,
    link,
    write_camera_sensor_yaml,
    write_ground_truth,
    write_imu_csv,
    write_imu_sensor_yaml,
)
from msckf_gates.common import large_data_dir

GROUPS = ("indoor_forward", "indoor_45", "outdoor_forward", "outdoor_45")
CAMERA_HZ = 30.0


def group_of(seq_dir: Path) -> str:
    for g in GROUPS:
        if seq_dir.name.startswith(g + "_"):
            return g
    raise ValueError(f"{seq_dir.name}: not a UZH-FPV sequence")


def calibration_dir(seq_dir: Path) -> Path:
    g = group_of(seq_dir)
    return seq_dir.parent / f"{g}_calib_snapdragon" / f"{g}_calib_snapdragon"


def load_camchain(calib_dir: Path) -> dict:
    (path,) = calib_dir.glob("camchain-imucam-*.yaml")
    return yaml.safe_load(path.read_text())


def write_camera(seq_dir: Path, cam: str, chain: dict) -> None:
    out = seq_dir / "mav0" / cam
    out.mkdir(parents=True, exist_ok=True)
    link(seq_dir / "img", out / "data")
    side = {"cam0": "left", "cam1": "right"}[cam]
    rows = np.loadtxt(seq_dir / f"{side}_images.txt", dtype=str, comments="#")
    with open(out / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for _, t, name in rows:
            f.write(f"{round(float(t) * 1e9)},{Path(name).name}\n")
    c = chain[cam]
    T_BS = np.linalg.inv(np.array(c["T_cam_imu"], dtype=np.float64))
    model = {
        "camera_model": c["camera_model"],
        "intrinsics": c["intrinsics"],
        "distortion_model": c["distortion_model"],
        "distortion_coefficients": c["distortion_coeffs"],
        "timeshift_cam_imu": c["timeshift_cam_imu"],
    }
    write_camera_sensor_yaml(out / "sensor.yaml", T_BS, c["resolution"], CAMERA_HZ, model,
                             f"UZH-FPV {seq_dir.name} {cam}, converted by python/uzh_fpv_to_euroc.py")  # fmt: skip


def write_imu(seq_dir: Path, calib_dir: Path) -> np.ndarray:
    out = seq_dir / "mav0" / "imu0"
    out.mkdir(parents=True, exist_ok=True)
    raw = np.loadtxt(seq_dir / "imu.txt", dtype=np.float64, comments="#")  # id t w a
    samples = np.column_stack([np.round(raw[:, 1] * 1e9), raw[:, 2:8]])
    write_imu_csv(samples, out / "data.csv")
    noise = yaml.safe_load((calib_dir / "imu.yaml").read_text())
    write_imu_sensor_yaml(out / "sensor.yaml", float(noise["update_rate"]), noise,
                          f"UZH-FPV Snapdragon IMU, noise from {calib_dir.name}/imu.yaml")  # fmt: skip
    return samples


def ground_truth(seq_dir: Path) -> np.ndarray:
    raw = np.loadtxt(seq_dir / "groundtruth.txt", dtype=np.float64, comments="#")  # t p q_xyzw
    gt = np.empty_like(raw)
    gt[:, 0] = np.round(raw[:, 0] * 1e9)
    gt[:, 1:4] = raw[:, 1:4]
    gt[:, 4] = raw[:, 7]
    gt[:, 5:8] = raw[:, 4:7]
    return gt


def convert(seq_dir: Path) -> None:
    calib_dir = calibration_dir(seq_dir)
    chain = load_camchain(calib_dir)
    for cam in ("cam0", "cam1"):
        write_camera(seq_dir, cam, chain)
    imu = write_imu(seq_dir, calib_dir)
    gt = ground_truth(seq_dir)
    write_ground_truth(gt, seq_dir / "gt_imu.csv")
    print(gravity_check_line(seq_dir.name, imu, gt))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", type=Path)
    args = ap.parse_args()
    seqs = args.sequences or sorted((large_data_dir() / "uzh_fpv").glob("*_snapdragon_with_gt"))
    if not seqs:
        raise SystemExit("no UZH-FPV sequences with ground truth")
    for s in seqs:
        convert(s)


if __name__ == "__main__":
    main()
