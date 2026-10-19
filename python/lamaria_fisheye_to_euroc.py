#!/usr/bin/env python3
"""Exports the raw fisheye SLAM cameras of LaMAria recordings (raw/<seq>.vrs,
from scripts/data/fetch_lamaria.sh -r) in the EuRoC layout, next to the
undistorted pinhole export that python/lamaria_to_euroc.py lays out:

  <seq>/fisheye/mav0/cam0/data.csv, data/<t_ns>.png  camera-slam-left, 640x480, 20 Hz
  <seq>/fisheye/mav0/cam1/...                        camera-slam-right
  <seq>/fisheye/mav0/camN/sensor.yaml                Aria's Fisheye624 calibration
                                                     (camera_model fisheye624,
                                                     distortion radtan-thinprism)
                                                     and T_BS from
                                                     aria_calibrations/<seq>.json
  <seq>/fisheye/mav0/imu0                            links to the pinhole export's IMU
  <seq>/fisheye/gt_imu.csv                           link to the pinhole export's truth

The body frame is the right IMU in both exports (the JSON's imu0 T_b_s is the
identity), so the IMU and the ground truth are shared. Reads the VRS with
projectaria_tools (Apache 2.0) and checks that the JSON's cam0 is the left
SLAM camera: T_b_cam0 must equal T_imu_device T_device_camera-slam-left.

    python python/lamaria_fisheye_to_euroc.py [SEQ_DIR...]

SEQ_DIR defaults to every <MSCKF_DATA_LARGE>/lamaria/training/* with a VRS.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

from euroc_layout import link, rigid, write_camera_sensor_yaml
from msckf_gates.common import large_data_dir

CAMERAS = {"cam0": ("camera-slam-left", "1201-1"), "cam1": ("camera-slam-right", "1201-2")}
CAMERA_HZ = 20.0


def check_camera_frame(provider, cam: str, calib: dict) -> None:
    """The JSON's T_b_s for cam must match the VRS device calibration."""
    device = provider.get_device_calibration()
    T_device_imu = device.get_transform_device_sensor("imu-right").to_matrix()
    T_device_cam = device.get_camera_calib(CAMERAS[cam][0]).get_transform_device_camera().to_matrix()
    T_imu_cam = np.linalg.inv(T_device_imu) @ T_device_cam
    T_json = rigid(calib[cam]["T_b_s"]["qvec"], calib[cam]["T_b_s"]["tvec"])
    if not np.allclose(T_imu_cam, T_json, atol=1e-6):
        raise RuntimeError(f"{cam} is not {CAMERAS[cam][0]}: T_b_s differs from the VRS calibration")


def write_sensor_yaml(out: Path, cam: str, calib: dict, seq_name: str) -> None:
    c = calib[cam]
    if c["model"] != "RAD_TAN_THIN_PRISM_FISHEYE" or len(c["params"]) != 16:
        raise RuntimeError(f"{seq_name} {cam}: expected RAD_TAN_THIN_PRISM_FISHEYE with 16 params")
    model = {
        "camera_model": "fisheye624",
        "intrinsics": c["params"][:4],
        "distortion_model": "radtan-thinprism",
        "distortion_coefficients": c["params"][4:],
    }
    T_BS = rigid(c["T_b_s"]["qvec"], c["T_b_s"]["tvec"])
    write_camera_sensor_yaml(out, T_BS, (c["resolution"]["width"], c["resolution"]["height"]), CAMERA_HZ, model,
                             f"LaMAria {seq_name} {CAMERAS[cam][0]} (Aria Fisheye624), "
                             "exported by python/lamaria_fisheye_to_euroc.py")  # fmt: skip


def export_images(provider, cam: str, out: Path) -> int:
    """Writes every frame of the camera's stream as <t_ns>.png; returns the count."""
    sid = StreamId(CAMERAS[cam][1])
    (out / "data").mkdir(parents=True, exist_ok=True)
    n = provider.get_num_data(sid)
    with open(out / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for i in range(n):
            image, record = provider.get_image_data_by_index(sid, i)
            t_ns = record.capture_timestamp_ns
            path = out / "data" / f"{t_ns}.png"
            if not path.exists() and not cv2.imwrite(str(path), image.to_numpy_array()):
                raise RuntimeError(f"cannot write {path}")
            f.write(f"{t_ns},{t_ns}.png\n")
    return n


def convert(seq_dir: Path) -> None:
    name = seq_dir.name
    calib = json.loads((seq_dir / "aria_calibrations" / f"{name}.json").read_text())
    pinhole = seq_dir / "mav0"
    if not (pinhole / "imu0" / "sensor.yaml").exists():
        raise RuntimeError(f"{name}: run python/lamaria_to_euroc.py first (IMU and ground truth)")
    provider = data_provider.create_vrs_data_provider(str(seq_dir / "raw" / f"{name}.vrs"))
    out = seq_dir / "fisheye"
    counts = []
    for cam in CAMERAS:
        check_camera_frame(provider, cam, calib)
        counts.append(export_images(provider, cam, out / "mav0" / cam))
        write_sensor_yaml(out / "mav0" / cam / "sensor.yaml", cam, calib, name)
    (out / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
    for f in ("data.csv", "sensor.yaml"):
        link(pinhole / "imu0" / f, out / "mav0" / "imu0" / f)
    if (seq_dir / "gt_imu.csv").exists():
        link(seq_dir / "gt_imu.csv", out / "gt_imu.csv")
    print(f"{name}: {counts[0]} left and {counts[1]} right fisheye images in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", type=Path)
    args = ap.parse_args()
    seqs = args.sequences or sorted(
        p.parent.parent for p in (large_data_dir() / "lamaria" / "training").glob("*/raw/*.vrs")
    )
    if not seqs:
        raise SystemExit("no LaMAria recordings (scripts/data/fetch_lamaria.sh -r)")
    for s in seqs:
        convert(s)


if __name__ == "__main__":
    main()
