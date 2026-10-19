#!/usr/bin/env python3
"""Writes the cross-check fixture for our Fisheye624 camera model
(tests/test_camera_aria_crosscheck.cpp): projections and unprojections of
Project Aria's own implementation (projectaria_tools, Apache 2.0) on a
LaMAria calibration.

  tests/data/fisheye624_aria.csv
    # params fx fy cx cy k0..k5 p0 p1 s0..s3 (16 values, fx = fy)
    # width height
    x,y,z,u,v,bx,by,bz     point in the camera frame, Aria's pixel, Aria's
                           unprojection of that pixel as a unit bearing

    python python/aria_fisheye_fixture.py [--calib aria_calibrations/<seq>.json] [--points N]

The calibration defaults to LaMAria R_01_easy cam0 (the camera test cases
quote the same values). Points are uniform directions up to 70 degrees from
the axis at random depths; only those Aria projects inside the image are kept.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3

from msckf_gates.common import REPO, large_data_dir

DEFAULT_CALIB = large_data_dir() / "lamaria" / "training" / "R_01_easy" / "aria_calibrations" / "R_01_easy.json"
MAX_ANGLE_DEG = 70.0


def aria_camera(params: list, width: int, height: int):
    """Aria's Fisheye624 has one focal length: params fx cx cy k0..k5 p0 p1 s0..s3."""
    if params[0] != params[1]:
        raise ValueError("Fisheye624 needs fx == fy")
    aria_params = np.array(params[:1] + params[2:])
    return calibration.CameraCalibration("cam", calibration.CameraModelType.FISHEYE624, aria_params,
                                         SE3(), width, height, None, np.pi, "")  # fmt: skip


def random_points(n: int, rng: np.random.Generator) -> np.ndarray:
    cos_max = np.cos(np.radians(MAX_ANGLE_DEG))
    z = rng.uniform(cos_max, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    s = np.sqrt(1.0 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z]) * rng.uniform(0.2, 20.0, n)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--points", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=REPO / "tests" / "data" / "fisheye624_aria.csv")
    args = ap.parse_args()
    cam0 = json.loads(args.calib.read_text())["cam0"]
    width, height = cam0["resolution"]["width"], cam0["resolution"]["height"]
    cam = aria_camera(cam0["params"], width, height)
    rng = np.random.default_rng(3)
    rows = []
    for p in random_points(args.points, rng):
        uv = cam.project(p)
        if uv is None:
            continue
        b = cam.unproject(uv)
        rows.append([*p, *uv, *(b / np.linalg.norm(b))])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# params " + " ".join(repr(float(x)) for x in cam0["params"]) + "\n")
        f.write(f"# {width} {height}\n")
        f.write("x,y,z,u,v,bx,by,bz\n")
        for r in rows:
            f.write(",".join(repr(float(x)) for x in r) + "\n")
    print(f"{len(rows)} samples -> {args.out}")


if __name__ == "__main__":
    main()
