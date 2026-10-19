#!/usr/bin/env python3
"""Writes a ROS1 bag from a sequence in EuRoC layout (mav0/camN, mav0/imu0),
for running the baseline binaries (OpenVINS, sqrtVINS: scripts/baselines)
on the same data as our gates. Pure Python (rosbags), no ROS needed.

    python python/euroc_to_rosbag.py --euroc DIR --out BAG [--cams cam0,cam1]
        [--cam-topics /cam0/image_raw,/cam1/image_raw] [--imu-topic /imu0]

Images go out as sensor_msgs/Image mono8 (colour images are converted), the
IMU as sensor_msgs/Imu, both stamped with the EuRoC timestamps. The topic
names must match the baseline's config (its camchain "rostopic" fields).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from msckf_gates.common import euroc_images

TYPES = get_typestore(Stores.ROS1_NOETIC)
Image = TYPES.types["sensor_msgs/msg/Image"]
Imu = TYPES.types["sensor_msgs/msg/Imu"]
Header = TYPES.types["std_msgs/msg/Header"]
Time = TYPES.types["builtin_interfaces/msg/Time"]
Vector3 = TYPES.types["geometry_msgs/msg/Vector3"]
Quaternion = TYPES.types["geometry_msgs/msg/Quaternion"]


def header(seq: int, t_ns: int, frame_id: str):
    return Header(seq=seq, stamp=Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000), frame_id=frame_id)


def image_message(seq: int, t_ns: int, path: Path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {path}")
    h, w = gray.shape
    return Image(header=header(seq, t_ns, "cam"), height=h, width=w, encoding="mono8", is_bigendian=0, step=w,
                 data=np.ascontiguousarray(gray).reshape(-1))  # fmt: skip


def imu_message(seq: int, row: np.ndarray):
    unknown = np.full(9, -1.0)
    return Imu(header=header(seq, int(row[0]), "imu"), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
               orientation_covariance=unknown, angular_velocity=Vector3(x=row[1], y=row[2], z=row[3]),
               angular_velocity_covariance=unknown, linear_acceleration=Vector3(x=row[4], y=row[5], z=row[6]),
               linear_acceleration_covariance=unknown)  # fmt: skip


def write_bag(euroc: Path, out: Path, cams: list[str], cam_topics: list[str], imu_topic: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    imu = np.loadtxt(euroc / "mav0" / "imu0" / "data.csv", delimiter=",", dtype=np.float64)
    with Writer(out) as bag:
        imu_conn = bag.add_connection(imu_topic, Imu.__msgtype__, typestore=TYPES)
        for i, row in enumerate(imu):
            bag.write(imu_conn, int(row[0]), TYPES.serialize_ros1(imu_message(i, row), Imu.__msgtype__))
        for cam, topic in zip(cams, cam_topics):
            conn = bag.add_connection(topic, Image.__msgtype__, typestore=TYPES)
            for i, (t_ns, path) in enumerate(euroc_images(euroc / "mav0" / cam)):
                bag.write(conn, t_ns, TYPES.serialize_ros1(image_message(i, t_ns, path), Image.__msgtype__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cams", default="cam0")
    ap.add_argument("--cam-topics", default="/cam0/image_raw")
    ap.add_argument("--imu-topic", default="/imu0")
    args = ap.parse_args()
    cams, topics = args.cams.split(","), args.cam_topics.split(",")
    if len(cams) != len(topics):
        raise SystemExit("--cams and --cam-topics need the same count")
    write_bag(args.euroc, args.out, cams, topics, args.imu_topic)
    print(f"{args.out}: {len(cams)} camera(s), {args.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
