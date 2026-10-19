"""EuRoC-layout writers shared by the dataset converters (LaMAria, UZH-FPV):
ground truth as TUM-VI's pose-only gt_imu.csv, IMU data.csv, sensor.yaml
files, relative symlinks, and the gravity check that validates frame
conventions (the mean world-frame specific force under the ground truth is
(0, 0, +g) when the poses are world_from_imu).
"""

import os
from pathlib import Path

import numpy as np
import yaml

from msckf_gates.common import quat_to_R


def rotation_xyzw(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return quat_to_R((w, x, y, z))


def rigid(q_xyzw, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation_xyzw(q_xyzw)
    T[:3, 3] = tvec
    return T


def link(target: Path, name: Path) -> None:
    """Relative symlink name -> target; existing links are kept."""
    if not name.is_symlink():
        os.symlink(os.path.relpath(target, name.parent), name)


def write_ground_truth(gt: np.ndarray, out: Path) -> None:
    """Rows 't_ns tx ty tz qw qx qy qz' of world_from_imu."""
    with open(out, "w") as f:
        f.write("# timestamp[ns],tx,ty,tz,qw,qx,qy,qz\n")
        for r in gt:
            f.write(f"{int(r[0])}," + ",".join(f"{x:.9f}" for x in r[1:]) + "\n")


def write_imu_csv(samples: np.ndarray, out: Path) -> None:
    """Rows 't_ns wx wy wz ax ay az' as EuRoC's imu0/data.csv."""
    with open(out, "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")  # fmt: skip
        for r in samples:
            f.write(f"{int(r[0])}," + ",".join(f"{x:.9f}" for x in r[1:]) + "\n")


def write_imu_sensor_yaml(out: Path, rate_hz: float, noise: dict, comment: str) -> None:
    """noise: gyroscope_noise_density, gyroscope_random_walk,
    accelerometer_noise_density, accelerometer_random_walk."""
    keys = ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
            "accelerometer_random_walk")  # fmt: skip
    sensor = {"sensor_type": "imu", "comment": comment, "rate_hz": rate_hz, **{k: float(noise[k]) for k in keys}}
    out.write_text(yaml.safe_dump(sensor, sort_keys=False))


def write_camera_sensor_yaml(out: Path, T_BS: np.ndarray, resolution, rate_hz: float, model: dict,
                             comment: str) -> None:  # fmt: skip
    """model: camera_model, intrinsics, distortion_model, distortion_coefficients,
    and optionally timeshift_cam_imu (Kalibr: t_imu = t_cam + shift)."""
    sensor = {
        "sensor_type": "camera",
        "comment": comment,
        "T_BS": {"cols": 4, "rows": 4, "data": [float(x) for x in T_BS.ravel()]},
        "rate_hz": rate_hz,
        "resolution": [int(x) for x in resolution],
        "camera_model": model["camera_model"],
        "intrinsics": [float(x) for x in model["intrinsics"]],
        "distortion_model": model["distortion_model"],
        "distortion_coefficients": [float(x) for x in model["distortion_coefficients"]],
    }
    if "timeshift_cam_imu" in model:
        sensor["timeshift_cam_imu"] = float(model["timeshift_cam_imu"])
    out.write_text(yaml.safe_dump(sensor, sort_keys=False))


def mean_world_specific_force(imu: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean over the ground-truth poses of R_wi times the mean accelerometer
    reading until the next pose. imu rows 't_ns w a', gt rows 't_ns p q_wxyz'."""
    k = np.searchsorted(gt[:, 0], imu[:, 0], side="right") - 1
    inside = (k >= 0) & (k < len(gt) - 1)
    k, acc = k[inside], imu[inside, 4:7]
    counts = np.bincount(k, minlength=len(gt))
    sums = np.stack([np.bincount(k, acc[:, i], minlength=len(gt)) for i in range(3)], axis=1)
    return np.mean([quat_to_R(gt[j, 4:8]) @ (sums[j] / counts[j]) for j in np.flatnonzero(counts)], axis=0)


def gravity_check_line(name: str, imu: np.ndarray, gt: np.ndarray) -> str:
    f = mean_world_specific_force(imu, gt)
    gap = np.diff(gt[:, 0]).max() * 1e-9
    return (f"{name}: {len(gt)} poses over {(gt[-1, 0] - gt[0, 0]) * 1e-9:.0f} s (max gap {gap:.2f} s), "
            f"mean world specific force ({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}) m/s^2")  # fmt: skip
