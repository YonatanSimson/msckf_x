#!/usr/bin/env python3
"""Converts KITTI raw drives (unsynced "extract" form, from
scripts/data/fetch_kitti_raw.sh) into the EuRoC layout our tools read:

  <out>/<drive>/mav0/cam0/data.csv     image_00 (left grey, raw, distorted)
  <out>/<drive>/mav0/cam0/data         link to image_00/data
  <out>/<drive>/mav0/cam0/sensor.yaml  K_00 / D_00 (pinhole radtan with k3),
                                       T_BS: camera pose in the IMU (OXTS) frame
  <out>/<drive>/mav0/imu0/data.csv     OXTS angular rates (wx, wy, wz) and
                                       accelerations (ax, ay, az; specific force,
                                       gravity included), 100 Hz
  <out>/<drive>/mav0/imu0/sensor.yaml  noise densities (see NOISE)
  <out>/<drive>/gt_imu.csv             OXTS state (GPS/INS), EuRoC layout: position from
                                       lat/lon/alt by the KITTI devkit's
                                       Mercator projection, relative to the first
                                       sample; orientation Rz(yaw) Ry(pitch) Rx(roll)
                                       (x forward, y left, z up; world z up),
                                       velocity (ve, vn, vu), biases 0

Timestamps out of order (OXTS jumps back by up to 40 ms) are dropped, keeping
a strictly increasing stream; gaps (up to 0.55 s) remain, and the dynamic
initializer refuses windows across them.

Extrinsic: T_cam0_imu = T_cam0_velo T_velo_imu (calib_velo_to_cam.txt,
calib_imu_to_velo.txt); T_BS = T_cam0_imu^-1. Timestamps are each sensor's own
(timestamps.txt), as nanoseconds.

    python python/kitti_raw_to_euroc.py [DRIVE_DIR...] [--out DIR]

DRIVE_DIR defaults to every <MSCKF_DATA_LARGE>/kitti/raw/*/*_extract.
"""

import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import yaml

from msckf_gates.common import large_data_dir, quat_wxyz

# OXTS RT3003 outputs are the unit's own filtered rates and accelerations, and
# its datasheet gives no white-noise densities for them. Effective densities
# measured by python/kitti_imu_consistency.py against the OXTS ground truth
# (attitude, fused velocity) on the five default drives, 1 s windows (upper
# bounds, as they include the ground truth's errors): gyro 1.74e-3
# rad/s/sqrt(Hz), accelerometer 1.22e-2 m/s^2/sqrt(Hz), about 10x and 6x
# EuRoC's ADIS16448 values; the bias walks are scaled by the same factors.
NOISE = {
    "gyroscope_noise_density": 1.74e-03,
    "gyroscope_random_walk": 1.9e-04,
    "accelerometer_noise_density": 1.22e-02,
    "accelerometer_random_walk": 1.8e-02,
}
EARTH_RADIUS = 6378137.0


def read_calib(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        key, _, val = line.partition(":")
        try:
            out[key.strip()] = np.array([float(x) for x in val.split()])
        except ValueError:
            pass  # calib_time
    return out


def rigid(R, T):
    M = np.eye(4)
    M[:3, :3] = np.asarray(R).reshape(3, 3)
    M[:3, 3] = T
    return M


def timestamps_ns(path):
    """'2011-09-26 14:01:22.271389602' -> ns (UTC; consistent across sensors)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        date, frac = line.strip().split(".")
        sec = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        out.append(int(sec) * 10**9 + int(frac.ljust(9, "0")[:9]))
    return np.array(out, dtype=np.int64)


def increasing(t):
    """Mask keeping a strictly increasing stream: each sample later than the
    last kept one (KITTI's OXTS timestamps jump back by up to 40 ms)."""
    keep = np.zeros(len(t), dtype=bool)
    last = None
    for i, x in enumerate(t):
        if last is None or x > last:
            keep[i] = True
            last = x
    return keep


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def convert(drive_dir: Path, out_root: Path):
    day_dir = drive_dir.parent
    name = drive_dir.name.replace("_extract", "")
    out = out_root / name / "mav0"
    for d in ("cam0", "imu0"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # Camera.
    cc = read_calib(day_dir / "calib_cam_to_cam.txt")
    K, D, S = cc["K_00"].reshape(3, 3), cc["D_00"], cc["S_00"]
    T_velo_imu = rigid(**{k: v for k, v in read_calib(day_dir / "calib_imu_to_velo.txt").items() if k in "RT"})
    T_cam_velo = rigid(**{k: v for k, v in read_calib(day_dir / "calib_velo_to_cam.txt").items() if k in "RT"})
    T_BS = np.linalg.inv(T_cam_velo @ T_velo_imu)
    sensor = {
        "sensor_type": "camera",
        "comment": f"KITTI raw {name} image_00, converted by python/kitti_raw_to_euroc.py",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(x) for x in T_BS.ravel()]},
        "rate_hz": 10,
        "resolution": [int(S[0]), int(S[1])],
        "camera_model": "pinhole",
        "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [float(x) for x in D],
    }
    (out / "cam0" / "sensor.yaml").write_text(yaml.safe_dump(sensor, sort_keys=False))
    t_img = timestamps_ns(drive_dir / "image_00" / "timestamps.txt")
    images = sorted((drive_dir / "image_00" / "data").glob("*.png"))
    if len(images) != len(t_img):
        raise SystemExit(f"{name}: {len(images)} images, {len(t_img)} timestamps")
    link = out / "cam0" / "data"
    if not link.exists():
        os.symlink(drive_dir / "image_00" / "data", link)
    keep = increasing(t_img)
    with open(out / "cam0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for t, p, k in zip(t_img, images, keep):
            if k:
                f.write(f"{t},{p.name}\n")

    # IMU and ground truth from the OXTS records.
    t_ox = timestamps_ns(drive_dir / "oxts" / "timestamps.txt")
    ox = np.array([np.loadtxt(p) for p in sorted((drive_dir / "oxts" / "data").glob("*.txt"))])
    if len(ox) != len(t_ox):
        raise SystemExit(f"{name}: {len(ox)} oxts records, {len(t_ox)} timestamps")
    keep = increasing(t_ox)
    t_ox, ox = t_ox[keep], ox[keep]
    with open(out / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")  # fmt: skip
        for t, r in zip(t_ox, ox):
            f.write(f"{t},{r[17]:.9g},{r[18]:.9g},{r[19]:.9g},{r[11]:.9g},{r[12]:.9g},{r[13]:.9g}\n")
    imu_yaml = {"sensor_type": "imu", "comment": "OXTS RT3003; noise: see NOISE in kitti_raw_to_euroc.py",
                "rate_hz": 100, **NOISE}  # fmt: skip
    (out / "imu0" / "sensor.yaml").write_text(yaml.safe_dump(imu_yaml, sort_keys=False))

    lat0 = ox[0, 0]
    scale = np.cos(np.deg2rad(lat0))

    def mercator(lat, lon, alt):
        x = scale * EARTH_RADIUS * np.deg2rad(lon)
        y = scale * EARTH_RADIUS * np.log(np.tan(np.deg2rad(90.0 + lat) / 2.0))
        return np.array([x, y, alt])

    p0 = mercator(*ox[0, :3])
    # EuRoC's state ground-truth layout: pose, world velocity (the OXTS's own
    # fused ve, vn, vu; differencing its positions disagrees with it by 0.2 m/s
    # median, 0.5-0.7 m/s p90, at any baseline), biases unknown (zero).
    with open(out.parent / "gt_imu.csv", "w") as f:
        f.write("# timestamp[ns],tx,ty,tz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz\n")
        for t, r in zip(t_ox, ox):
            p = mercator(*r[:3]) - p0
            q = quat_wxyz(rot(r[3], r[4], r[5]))
            f.write(f"{t},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},{q[3]:.9f},"
                    f"{r[7]:.6f},{r[6]:.6f},{r[10]:.6f},0,0,0,0,0,0\n")  # fmt: skip
    print(f"{name}: {keep.sum()} IMU samples, {len(images)} images, "
          f"{(t_ox[-1] - t_ox[0]) * 1e-9:.1f} s -> {out.parent}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=large_data_dir() / "kitti" / "euroc")
    args = ap.parse_args()
    drives = args.drives or sorted((large_data_dir() / "kitti" / "raw").glob("*/*_extract"))
    if not drives:
        raise SystemExit("no drives (scripts/data/fetch_kitti_raw.sh)")
    for d in drives:
        convert(d, args.out)


if __name__ == "__main__":
    main()
