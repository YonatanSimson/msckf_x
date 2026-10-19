#!/usr/bin/env python3
"""Plays back dynamic initialization (Stage 1b) on a TUM-VI sequence, with
the estimated and true gravity drawn as plumb lines in the image.

Runs the adopted front end (tracker_eval tracks) and the pipeline with retry
(dynamic_init_eval euroc, the Stage 1b gate's settings, plus --restart:
after a failure it starts again every --restart s), cached in
<out>/<sequence>_<cam>/ (--rerun to recompute), then shows every frame of
every pipeline run from start time --t0 (the gate's start labels: seconds
after the first frame + 0.5 s) from trace.csv:

  plumb lines  from the principal point along gravity, i.e. the image of the
               3D line {c + s g_C, s >= 0} with c on the optical axis at
               --depth m; yellow: the estimate, green: ground truth
  dots         tracked features of the frame
  text         the attempt (start, window, outcome and why), the gravity
               error, estimated and true speed

The estimate is the hand-off (static or refined) or, for a refused attempt,
the linear solution when it has one, carried to each frame by IMU
preintegration (TraceLog in tools/dynamic_init_eval.cpp). Frames without an
estimate show the truth only.

Keys (imshow window): space or a left click pause / resume, n step while
paused, q / Esc quit.

    python python/play_init.py room1 [--cam cam1] [--t0 0] [--restart 10]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from msckf_gates.common import FRONT_END_TUMVI, REPO, data_dir, euroc_images, fnum, read_csv, run, tool
from msckf_gates.sequences import REAL_NOISE
from play_euroc import to_u8
from viz import FrameSink

YELLOW, GREEN, WHITE, GREY = (0, 220, 255), (0, 200, 0), (255, 255, 255), (150, 150, 150)


def plumb_line(g_c: np.ndarray, K: np.ndarray, D: np.ndarray, depth: float, length: float) -> np.ndarray:
    """Pixels of the line from the optical axis point at `depth` along the
    gravity direction g_c (camera frame), equidistant fisheye."""
    s = np.linspace(0.0, length, 30)[:, None]
    pts = np.array([0.0, 0.0, depth]) + s * g_c[None, :]
    pts = pts[pts[:, 2] > 0.05]
    uv, _ = cv2.fisheye.projectPoints(pts[None].astype(np.float64), np.zeros(3), np.zeros(3), K, D)
    return np.asarray(uv[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="TUM-VI name, e.g. room1")
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--t0", type=float, default=0.0, help="start label (s), as in the gate's runs.csv")
    ap.add_argument("--restart", type=float, default=10.0, help="restart period after a failure (s)")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--depth", type=float, default=1.0, help="plumb line anchor depth on the optical axis (m)")
    ap.add_argument("--length", type=float, default=3.0, help="plumb line length (m)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=1.5, help="display scale")
    ap.add_argument("--out", type=Path, default=REPO / "init_viz_out")
    ap.add_argument("--rerun", action="store_true", help="re-run tracker and initializer even if cached")
    ap.add_argument("--video", action="store_true", help="write a video instead of opening a window")
    args = ap.parse_args()

    root = args.tumvi / f"dataset-{args.sequence}_512_16"
    camchain = root / "dso" / "camchain.yaml"
    if not (root / "mav0" / args.cam).exists():
        raise SystemExit(f"{root}/mav0/{args.cam} not found (scripts/data/fetch_tumvi.sh {args.sequence})")
    d = args.out / f"{args.sequence}_{args.cam}"
    tracks = d / "tracks" / "tracks.csv"
    if args.rerun or not tracks.exists():
        run([tool("tracker_eval"), "tracks", "--euroc", root, "--cam", args.cam, "--camchain", camchain,
             *FRONT_END_TUMVI, "--out", tracks.parent], d / "tracks.log")  # fmt: skip
    if args.rerun or not (d / "trace.csv").exists():
        imu_yaml = args.tumvi / "kalibr_results" / "pinhole-equi-512" / "imu-imucalib.yaml"
        run([tool("dynamic_init_eval"), "euroc", "--euroc", root, "--cam", args.cam, "--camchain", camchain,
             "--gt", root / "dso" / "gt_imu.csv", "--tracks", tracks, "--imu-yaml", imu_yaml, "--bias", "zero",
             *REAL_NOISE, "--windows", "0.5", "--restart", args.restart, "--out", d], d / "init.log")  # fmt: skip

    cam = yaml.safe_load(camchain.read_text())[args.cam]
    fx, fy, cx, cy = cam["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array(cam["distortion_coeffs"], dtype=np.float64)

    rows = [r for r in read_csv(d / "trace.csv") if abs(fnum(r["t0"]) - args.t0) < 1e-6]
    if not rows:
        starts = sorted({r["t0"] for r in read_csv(d / "trace.csv")}, key=float)
        raise SystemExit(f"no run starts at --t0 {args.t0:g}; starts: {', '.join(starts)}")
    runs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        runs[r["run"]].append(r)

    images = euroc_images(root / "mav0" / args.cam)
    t_img = np.array([t for t, _ in images], dtype=np.int64)
    a = np.loadtxt(tracks, delimiter=",", skiprows=1, ndmin=2)
    feats: dict[int, np.ndarray] = {int(t): a[a[:, 1].astype(np.int64) == t, 3:5] for t in np.unique(a[:, 1])}

    s = args.scale
    centre = (int(round(cx * s)), int(round(cy * s)))
    with FrameSink(f"init_{args.sequence}_{args.cam}_t{args.t0:g}", args.fps, str(args.out), args.video) as sink:
        for run_rows in runs.values():
            r0 = run_rows[0]
            outcome = f"OK ({r0['init']})" if r0["ok"] == "1" else f"REFUSED: {r0['reason']}"
            t_first = fnum(r0["t"])
            start = args.t0 + fnum(r0["attempt_start"]) - fnum(runs[next(iter(runs))][0]["attempt_start"])
            print(f"{r0['method']} from {start:.1f} s: window {fnum(r0['window']):g} s, {outcome}")
            for r in run_rows:
                t_ns = int(round(fnum(r["t"]) * 1e9))
                k = int(np.argmin(np.abs(t_img - t_ns)))
                img = to_u8(cv2.imread(str(images[k][1]), cv2.IMREAD_UNCHANGED))
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
                vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                for u, v in feats.get(int(t_img[k]), np.empty((0, 2))):
                    cv2.circle(vis, (int(round(u * s)), int(round(v * s))), 2, GREY, -1, cv2.LINE_AA)
                g_est = np.array([fnum(r[f"g{c}_est"]) for c in "xyz"])
                g_true = np.array([fnum(r[f"g{c}_true"]) for c in "xyz"])
                err = np.nan
                for g, col in ((g_true, GREEN), (g_est, YELLOW)):
                    if np.isfinite(g).all():
                        pts = np.round(plumb_line(g, K, D, args.depth, args.length) * s).astype(np.int32)
                        cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)
                if np.isfinite(g_est).all() and np.isfinite(g_true).all():
                    err = np.degrees(np.arctan2(np.linalg.norm(np.cross(g_est, g_true)), g_est @ g_true))
                cv2.drawMarker(vis, centre, WHITE, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
                lines = [
                    f"{args.cam} {r['method']} from {start:.1f} s  frame {fnum(r['t']) - t_first:+.2f} s of a "
                    f"{fnum(r['window']):g} s window  {outcome}",
                    f"gravity (yellow est, green true) error {err:.2f} deg" if np.isfinite(err)
                    else "gravity: no estimate (green: true)",
                    f"speed est {fnum(r['speed_est']):.3f} m/s  true {fnum(r['speed_true']):.3f} m/s  "
                    f"features {r['features']}",
                ]  # fmt: skip
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
                for i, line in enumerate(lines):
                    cv2.putText(vis, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                if not sink.show(vis):
                    return


if __name__ == "__main__":
    main()
