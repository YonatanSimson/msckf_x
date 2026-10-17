#!/usr/bin/env python3
"""Adds the EuRoC layout our tools read to LaMAria training sequences (from
scripts/data/fetch_lamaria.sh), next to the downloaded aria/ folder:

  <seq>/mav0/camN/data.csv, data  links to aria/camN (pinhole, undistorted, 20 Hz)
  <seq>/mav0/camN/sensor.yaml     pinhole, zero radtan distortion, T_BS from
                                  pinhole_calibrations/<seq>.json
  <seq>/mav0/imu0/data.csv        link to aria/imu0 (Aria right IMU, 1 kHz, raw:
                                  the factory IMU calibration is not applied)
  <seq>/mav0/imu0/sensor.yaml     noise densities from the same JSON
  <seq>/gt_imu.csv                pseudo-dense ground truth (pGT) as world_from_imu,
                                  pose-only TUM-VI gt_imu.csv layout, without
                                  frozen (repeated) poses; only when the
                                  sequence has pGT

Conventions (lamaria/utils/aria.py): the body frame is the right IMU, so the
JSON's T_b_s is T_imu_cam; JSON qvec and pGT quaternions are x, y, z, w
(pycolmap). pGT comes in two kinds: MPS (R_01-R_10: world_from_imu at every
frame) and the control-point pipeline's (the sequences with control points:
world_from_cam0 at keyframes 0.1-0.65 s apart); both worlds are z up. Prints
the mean world-frame specific force under the ground truth, (0, 0, +g) when
the frames are right: with the control-point pGT read as world_from_imu it is
(-0.8, -1.1, 1.8) m/s^2 on R_11_5cp, as world_from_cam0 (0.1, -0.2, 9.8).

    python python/lamaria_to_euroc.py [SEQ_DIR...]

SEQ_DIR defaults to every <MSCKF_DATA_LARGE>/lamaria/training/*.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from msckf_gates.common import large_data_dir, quat_to_R, quat_wxyz


def rotation(q_xyzw):
    x, y, z, w = q_xyzw
    return quat_to_R((w, x, y, z))


def rigid(q_xyzw, tvec):
    T = np.eye(4)
    T[:3, :3] = rotation(q_xyzw)
    T[:3, 3] = tvec
    return T


def link(target, name):
    if not name.is_symlink():
        os.symlink(os.path.relpath(target, name.parent), name)


def write_camera(seq_dir, cam, calib):
    out = seq_dir / "mav0" / cam
    out.mkdir(parents=True, exist_ok=True)
    for f in ("data.csv", "data"):
        link(seq_dir / "aria" / cam / f, out / f)
    c = calib[cam]
    T_BS = rigid(c["T_b_s"]["qvec"], c["T_b_s"]["tvec"])
    sensor = {
        "sensor_type": "camera",
        "comment": f"LaMAria {seq_dir.name} {cam}, converted by python/lamaria_to_euroc.py",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(x) for x in T_BS.ravel()]},
        "rate_hz": 20,
        "resolution": [c["resolution"]["width"], c["resolution"]["height"]],
        "camera_model": "pinhole",
        "intrinsics": [float(x) for x in c["params"]],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    (out / "sensor.yaml").write_text(yaml.safe_dump(sensor, sort_keys=False))


def write_imu(seq_dir, imu):
    out = seq_dir / "mav0" / "imu0"
    out.mkdir(parents=True, exist_ok=True)
    link(seq_dir / "aria" / "imu0" / "data.csv", out / "data.csv")
    sensor = {
        "sensor_type": "imu",
        "comment": "Aria right IMU (raw); noise from pinhole_calibrations JSON",
        "rate_hz": imu["imu_rate"],
        "gyroscope_noise_density": imu["gyro_noise_density"],
        "gyroscope_random_walk": imu["gyro_bias_random_walk_sigma"],
        "accelerometer_noise_density": imu["acc_noise_density"],
        "accelerometer_random_walk": imu["acc_bias_random_walk_sigma"],
    }
    (out / "sensor.yaml").write_text(yaml.safe_dump(sensor, sort_keys=False))


def drop_frozen_poses(pgt):
    """pGT rows whose pose differs from both neighbours. MPS repeats its first
    pose for the first ~0.8 s (16-18 rows) while the head already turns, and
    holds a pose for a row or two mid-sequence: those rows are not truth."""
    pose = pgt[:, 1:]
    same = np.all(pose[1:] == pose[:-1], axis=1)
    frozen = np.zeros(len(pgt), dtype=bool)
    frozen[1:] |= same
    frozen[:-1] |= same
    return pgt[~frozen]


def imu_poses(pgt, T_rig_imu):
    """pGT rows 't_ns tx ty tz qx qy qz qw' (world_from_rig) -> rows
    't_ns tx ty tz qw qx qy qz' of world_from_imu."""
    out = np.empty_like(pgt)
    out[:, 0] = pgt[:, 0]
    for k, r in enumerate(pgt):
        T_wi = rigid(r[4:8], r[1:4]) @ T_rig_imu
        out[k, 1:4] = T_wi[:3, 3]
        out[k, 4:8] = quat_wxyz(T_wi[:3, :3])
    return out


def write_ground_truth(gt, out):
    with open(out, "w") as f:
        f.write("# timestamp[ns],tx,ty,tz,qw,qx,qy,qz\n")
        for r in gt:
            f.write(f"{int(r[0])}," + ",".join(f"{x:.9f}" for x in r[1:]) + "\n")


def mean_world_specific_force(imu, gt):
    """Mean over the ground-truth poses of R_wi times the mean accelerometer
    reading until the next pose."""
    k = np.searchsorted(gt[:, 0], imu[:, 0], side="right") - 1
    inside = (k >= 0) & (k < len(gt) - 1)
    k, acc = k[inside], imu[inside, 4:7]
    counts = np.bincount(k, minlength=len(gt))
    sums = np.stack([np.bincount(k, acc[:, i], minlength=len(gt)) for i in range(3)], axis=1)
    return np.mean([quat_to_R(gt[j, 4:8]) @ (sums[j] / counts[j]) for j in np.flatnonzero(counts)], axis=0)


def pgt_rig_from_imu(seq_dir, calib):
    """T_rig_imu of the sequence's pGT. MPS pGT (no control points) is
    world_from_imu; the control-point pipeline's pGT is world_from_cam0."""
    if not (seq_dir / "ground_truth" / "control_points" / f"{seq_dir.name}.json").exists():
        return np.eye(4)
    T_imu_cam0 = rigid(calib["cam0"]["T_b_s"]["qvec"], calib["cam0"]["T_b_s"]["tvec"])
    return np.linalg.inv(T_imu_cam0)


def convert(seq_dir):
    name = seq_dir.name
    calib = json.loads((seq_dir / "pinhole_calibrations" / f"{name}.json").read_text())
    for cam in ("cam0", "cam1"):
        write_camera(seq_dir, cam, calib)
    write_imu(seq_dir, calib["imu0"])
    pgt_path = seq_dir / "ground_truth" / "pGT" / f"{name}.txt"
    if not pgt_path.exists():
        print(f"{name}: no pGT, cameras and IMU only")
        return
    pgt = np.loadtxt(pgt_path, dtype=np.float64)
    gt = imu_poses(drop_frozen_poses(pgt), pgt_rig_from_imu(seq_dir, calib))
    write_ground_truth(gt, seq_dir / "gt_imu.csv")
    imu = np.loadtxt(seq_dir / "aria" / "imu0" / "data.csv", delimiter=",", dtype=np.float64)
    f = mean_world_specific_force(imu, gt)
    gap = np.diff(gt[:, 0]).max() * 1e-9
    print(f"{name}: {len(gt)} poses over {(gt[-1, 0] - gt[0, 0]) * 1e-9:.0f} s (max gap {gap:.2f} s), "
          f"mean world specific force ({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}) m/s^2")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", type=Path)
    args = ap.parse_args()
    seqs = args.sequences or sorted(p.parent for p in (large_data_dir() / "lamaria" / "training").glob("*/aria"))
    if not seqs:
        raise SystemExit("no sequences (scripts/data/fetch_lamaria.sh)")
    for s in seqs:
        convert(s)


if __name__ == "__main__":
    main()
