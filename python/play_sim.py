#!/usr/bin/env python3
"""Plays a simulated run in Rerun: the motion in 3D, what the camera sees, and the IMU.

Runs build/sim_export into gate_out/sim_export/<scenario>_seed<N> (or reads
--dir, a sim_export output), then logs on one "time" timeline:

  3D view       the landmark map (grey), the ground-truth path, the IMU body
                axes and the camera frustum; the landmarks the camera sees now
                in their track colour, with rays from the camera
  Camera view   the observed (noisy) pixels with a trail of their last
                --trail positions; the frustum uses the pinhole part of the
                camera model, the pixels include its distortion
  Time series   measured gyroscope and accelerometer, true gyro and
                accelerometer biases

The Rerun viewer opens (under WSL with DISPLAY=:0); --save writes an .rrd file
instead, to open later with `rerun FILE`.

    python python/play_sim.py [--scenario figure8] [--duration 60] [--seed 1] [--save run.rrd]
    python python/play_sim.py --dir gate_out/sim_export/figure8_seed1
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml

from msckf_gates.common import gate_out, run, tool

RGB = np.ndarray  # (n, 3) uint8 colours
XYZ = "xyz"


def export(args: argparse.Namespace) -> Path:
    out: Path = gate_out("sim_export") / f"{args.scenario}_seed{args.seed}"
    cmd = [tool("sim_export"), "--scenario", args.scenario, "--duration", args.duration, "--seed", args.seed]
    run(cmd + ["--out", out])
    return out


def load(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def track_colours(ids: np.ndarray) -> RGB:
    """A stable colour per landmark id (golden-ratio hues)."""
    hue = (ids * 0.618033988749895) % 1.0
    k = (np.array([5.0, 3.0, 1.0])[None, :] + hue[:, None] * 6.0) % 6.0
    rgb = 1.0 - 0.75 * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def xyzw(q_wxyz: np.ndarray) -> list[float]:
    return [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]


def rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q_wxyz[0], q_wxyz[1:]
    return np.asarray(v + 2.0 * np.cross(u, np.cross(u, v) + w * v))


def blueprint() -> rrb.Blueprint:
    series = rrb.Vertical(
        rrb.TimeSeriesView(origin="imu/gyro", name="Gyroscope (rad/s)"),
        rrb.TimeSeriesView(origin="imu/acc", name="Accelerometer (m/s²)"),
        rrb.Tabs(
            rrb.TimeSeriesView(origin="truth/gyro_bias", name="Gyro bias (rad/s)"),
            rrb.TimeSeriesView(origin="truth/acc_bias", name="Accelerometer bias (m/s²)"),
        ),
    )
    camera = rrb.Spatial2DView(origin="world/imu/cam", name="Camera")
    return rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="world", name="3D"), rrb.Vertical(camera, series)))


def log_static(config: dict, truth: np.ndarray, landmarks: np.ndarray) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/map", rr.Points3D(landmarks[:, 1:4], colors=[150, 150, 150], radii=0.015), static=True)
    rr.log("world/path", rr.LineStrips3D([truth[:, 1:4]], colors=[90, 160, 255], radii=0.01), static=True)
    cam = config["camera"]
    fx, fy, cx, cy = cam["params"][:4]
    rr.log("world/imu", rr.TransformAxes3D(0.3), static=True)
    q_ic = np.asarray(config["q_IC"], dtype=float)
    rr.log(
        "world/imu/cam",
        rr.Transform3D(translation=config["p_IC"], quaternion=rr.Quaternion(xyzw=xyzw(q_ic))),
        static=True,
    )
    pinhole = rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy], resolution=[cam["width"], cam["height"]],
                         camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.4)  # fmt: skip
    rr.log("world/imu/cam", pinhole, static=True)


def log_series(name: str, t: np.ndarray, values: np.ndarray) -> None:
    rr.log(name, rr.SeriesLines(names=list(XYZ), colors=[[230, 70, 70], [70, 190, 70], [70, 110, 240]]), static=True)
    rr.send_columns(name, indexes=[rr.TimeColumn("time", duration=t)], columns=rr.Scalars.columns(scalars=values))


def log_imu(imu: np.ndarray, truth: np.ndarray) -> None:
    log_series("imu/gyro", imu[:, 0], imu[:, 1:4])
    log_series("imu/acc", imu[:, 0], imu[:, 4:7])
    log_series("truth/gyro_bias", truth[:, 0], truth[:, 11:14])
    log_series("truth/acc_bias", truth[:, 0], truth[:, 14:17])


class FrameLogger:
    """Logs one camera frame: the body pose, the observed landmarks and the pixel trails."""

    def __init__(self, config: dict, truth: np.ndarray, landmarks: np.ndarray, trail: int) -> None:
        self.truth = truth
        self.time_offset: float = float(config["time_offset"])
        self.p_ic: np.ndarray = np.asarray(config["p_IC"], dtype=float)
        self.positions: dict[int, np.ndarray] = {int(i): p for i, p in zip(landmarks[:, 0], landmarks[:, 1:4])}
        self.trails: dict[int, deque] = {}
        self.trail = trail

    def log(self, t_cam: float, obs: np.ndarray) -> None:
        rr.set_time("time", duration=t_cam)
        k = int(np.searchsorted(self.truth[:, 0], t_cam + self.time_offset))
        row = self.truth[min(k, len(self.truth) - 1)]
        q = row[4:8]
        rr.log("world/imu", rr.Transform3D(translation=row[1:4], quaternion=rr.Quaternion(xyzw=xyzw(q))))
        ids = obs[:, 1].astype(int)
        colours = track_colours(ids)
        points = np.array([self.positions[i] for i in ids]).reshape(-1, 3)
        centre = row[1:4] + rotate(q, self.p_ic)
        rr.log("world/seen", rr.Points3D(points, colors=colours, radii=0.04))
        rays = [np.stack([centre, p]) for p in points]
        rr.log("world/rays", rr.LineStrips3D(rays, colors=np.c_[colours, np.full(len(ids), 60)], radii=0.003))
        rr.log("world/imu/cam/pixels", rr.Points2D(obs[:, 2:4], colors=colours, radii=2.5))
        rr.log("world/imu/cam/trails", rr.LineStrips2D(self.update_trails(ids, obs[:, 2:4]), colors=colours))

    def update_trails(self, ids: np.ndarray, uv: np.ndarray) -> list[np.ndarray]:
        self.trails = {i: self.trails.get(i, deque(maxlen=self.trail)) for i in ids}
        for i, p in zip(ids, uv):
            self.trails[i].append(p)
        return [np.array(self.trails[i]) for i in ids]


def play(directory: Path, trail: int) -> None:
    config: dict = yaml.safe_load((directory / "config.yaml").read_text())
    truth, landmarks = load(directory / "groundtruth.csv"), load(directory / "landmarks.csv")
    log_static(config, truth, landmarks)
    log_imu(load(directory / "imu.csv"), truth)
    features = load(directory / "features.csv")
    frames = FrameLogger(config, truth, landmarks, trail)
    starts = np.flatnonzero(np.r_[True, np.diff(features[:, 0]) != 0])
    for obs in np.split(features, starts[1:]):
        frames.log(float(obs[0, 0]), obs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="figure8", help="handheld, cart, fast_rotation, car or figure8")
    ap.add_argument("--duration", type=float, default=60.0, help="s")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dir", type=Path, help="a sim_export output to play instead of simulating")
    ap.add_argument("--trail", type=int, default=10, help="pixel trail length, in frames")
    ap.add_argument("--save", type=Path, help="write an .rrd file instead of opening the viewer")
    args = ap.parse_args()

    directory: Path = args.dir if args.dir is not None else export(args)
    rr.init("msckf_sim", default_blueprint=blueprint())
    if args.save is not None:
        rr.save(args.save)
    else:
        rr.spawn(executable_path=str(Path(sys.executable).parent / "rerun"))  # the venv's viewer, not on PATH
    play(directory, args.trail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
