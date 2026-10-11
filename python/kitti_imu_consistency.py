#!/usr/bin/env python3
"""Effective noise of the KITTI OXTS IMU, from its consistency with the OXTS
ground truth (python/kitti_raw_to_euroc.py output). The OXTS outputs are the
unit's own filtered rates and accelerations, and its datasheet gives no
white-noise densities for them.

Over 1 s windows without timestamp gaps:
  gyro: angle between the gyro-integrated rotation and the ground-truth
        attitude change
  accelerometer: |integrated (R a + g) - ground-truth velocity change|
        (ground-truth velocity: the OXTS's fused velocity)
Each pooled median over the drives is converted to a per-axis density, as
the median of a 3-dof chi distribution is ~1.54 sigma: density ~ median /
1.54 for a 1 s window. These include the ground truth's own errors, so they
are an upper bound on the IMU noise (conservative).

    python python/kitti_imu_consistency.py [DRIVE_DIR...]

DRIVE_DIR: converted drives (default: every $MSCKF_DATA_LARGE/kitti/euroc/*).
"""

import argparse
from pathlib import Path

import numpy as np

from msckf_gates.common import large_data_dir

G = np.array([0.0, 0.0, -9.81])
WINDOW = 100  # samples at 100 Hz: 1 s
CHI3_MEDIAN = 1.54


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def drive_errors(d: Path):
    imu = np.loadtxt(d / "mav0" / "imu0" / "data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(d / "gt_imu.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    dt = np.diff(t)
    R = [quat_to_R(q) for q in gt[:, 4:8]]
    f_world = np.array([Ri @ a for Ri, a in zip(R, imu[:, 4:7])])

    def vel(j, h=10):
        # The OXTS's fused velocity (columns 8-10) when converted with it;
        # differencing its positions is off by ~0.2 m/s.
        if gt.shape[1] >= 11:
            return gt[j, 8:11]
        return (gt[j + h, 1:4] - gt[j - h, 1:4]) / (t[j + h] - t[j - h])

    gyro, acc = [], []
    for i in range(10, len(t) - WINDOW - 11, 25):
        if dt[i - 10 : i + WINDOW + 10].max() > 0.015:
            continue
        Ri = np.eye(3)
        for j in range(i, i + WINDOW):
            Ri = Ri @ so3_exp(imu[j, 1:4] * dt[j])
        D = (R[i].T @ R[i + WINDOW]).T @ Ri
        gyro.append(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1)))
        dv = np.sum((f_world[i : i + WINDOW] + G) * dt[i : i + WINDOW, None], axis=0)
        acc.append(np.linalg.norm(dv - (vel(i + WINDOW) - vel(i))))
    return np.array(gyro), np.array(acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="*", type=Path)
    args = ap.parse_args()
    drives = args.drives or sorted(p for p in (large_data_dir() / "kitti" / "euroc").glob("*_drive_*"))
    all_g, all_a = [], []
    for d in drives:
        g, a = drive_errors(d)
        all_g.append(g)
        all_a.append(a)
        print(f"{d.name}: {len(g)} windows; 1 s gyro error median {np.median(g):.5f} rad, "
              f"velocity-change error median {np.median(a):.3f} m/s")  # fmt: skip
    g, a = np.concatenate(all_g), np.concatenate(all_a)
    print(f"pooled: gyro density ~ {np.median(g) / CHI3_MEDIAN:.2e} rad/s/sqrt(Hz), "
          f"accelerometer density ~ {np.median(a) / CHI3_MEDIAN:.2e} m/s^2/sqrt(Hz)")  # fmt: skip


if __name__ == "__main__":
    main()
