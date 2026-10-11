#!/usr/bin/env python3
"""Runs a tracker on a TUM-VI recording and plays it back with the tracks.

The tracker is our vendored Basalt flow (`tracker_eval tracks --tracker
basalt`, Basalt's shipped TUM-VI settings: frame_to_frame, 40 px grid, 0.2 px
forward-backward) unless --tracker klt. Tracks are cached in
<out>/<sequence>_<cam>_<tracker>/tracks.csv (--rerun to recompute), then
drawn on every frame:

  dot      live feature, coloured by age: red when new, through yellow, to
           green at --full-age frames and beyond
  trail    its last --trail positions
  x        a feature lost in this frame (where it was last)

Header: frame, live features, new, lost, median age. Keys (imshow window):
space pause, n step while paused, q / Esc quit.

    python python/play_tracks.py room1 [--cam cam1] [--tracker klt] [--start 500]

The sequence is a TUM-VI name (room1, calib-imu1, ...: ~/data/tumvi/
dataset-<name>_512_16, from scripts/data/fetch_tumvi.sh) or a path to a
EuRoC-format directory. Uses the published pinhole-equi calibration.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from msckf_gates.common import REPO, data_dir, run, tool
from play_euroc import to_u8
from tracker_compare import BASALT_REPO, euroc_images, shipped_flow_config, vendored_flow_args
from viz import FrameSink


def age_colour(age: int, full: int) -> tuple[int, int, int]:
    """BGR: red (new) -> yellow -> green (age >= full)."""
    t = min(age / max(full, 1), 1.0)
    return (0, int(255 * min(1.0, 2 * t)), int(255 * min(1.0, 2 * (1 - t))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="TUM-VI name (room1, calib-imu1, ...) or EuRoC-format directory")
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--tracker", choices=["basalt", "klt"], default="basalt")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--start", type=int, default=0, help="first frame shown")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--trail", type=int, default=10, help="trail length, frames")
    ap.add_argument("--full-age", type=int, default=20, help="age drawn fully green, frames")
    ap.add_argument("--scale", type=float, default=1.5, help="display scale")
    ap.add_argument("--out", type=Path, default=REPO / "tracker_viz_out")
    ap.add_argument("--rerun", action="store_true", help="re-run the tracker even if tracks are cached")
    ap.add_argument("--video", action="store_true", help="write a video instead of opening a window")
    args = ap.parse_args()

    root = Path(args.sequence).expanduser()
    if not (root / "mav0").exists():
        root = args.tumvi / f"dataset-{args.sequence}_512_16"
    if not (root / "mav0" / args.cam).exists():
        raise SystemExit(f"{root}/mav0/{args.cam} not found (scripts/data/fetch_tumvi.sh {args.sequence})")
    camchain = args.tumvi / "kalibr_results" / "pinhole-equi-512" / "camchain-imucam-imucalib.yaml"
    name = f"{root.name}_{args.cam}_{args.tracker}"
    d = args.out / name

    if args.rerun or not (d / "tracks.csv").exists():
        extra = (
            vendored_flow_args(shipped_flow_config(BASALT_REPO / "data" / "tumvi_512_config.json"))
            if args.tracker == "basalt"
            else []
        )
        run(
            [tool("tracker_eval"), "tracks", "--euroc", root, "--cam", args.cam, "--camchain", camchain, *extra,
             "--out", d],
            d.with_suffix(".log"),
        )  # fmt: skip

    a = np.loadtxt(d / "tracks.csv", delimiter=",", skiprows=1, ndmin=2)
    frame, track, uv = a[:, 0].astype(int), a[:, 2].astype(np.int64), a[:, 3:5]
    order = np.argsort(frame, kind="stable")
    images = euroc_images(root / "mav0" / args.cam)
    bounds = np.searchsorted(frame[order], np.arange(len(images) + 1))

    history: dict[int, list[tuple[float, float]]] = {}  # track -> positions, oldest first
    born: dict[int, int] = {}
    prev: dict[int, tuple[float, float]] = {}
    s = args.scale
    with FrameSink(name, args.fps, str(args.out), args.video) as sink:
        for k, (_, path) in enumerate(images):
            idx = order[bounds[k] : bounds[k + 1]]
            cur = {int(t): (float(p[0]), float(p[1])) for t, p in zip(track[idx], uv[idx])}
            for t, p in cur.items():
                born.setdefault(t, k)
                history.setdefault(t, []).append(p)
                del history[t][: -args.trail]
            lost = {t: p for t, p in prev.items() if t not in cur}
            for t in lost:
                history.pop(t, None)
            new = sum(born[t] == k for t in cur)
            prev = cur
            if k < args.start:
                continue

            img = to_u8(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            for t, (u, v) in cur.items():
                col = age_colour(k - born[t], args.full_age)
                pts = np.round(np.array(history[t]) * s).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
                cv2.circle(vis, (int(round(u * s)), int(round(v * s))), 3, col, -1, cv2.LINE_AA)
            for u, v in lost.values():
                c = (int(round(u * s)), int(round(v * s)))
                cv2.drawMarker(vis, c, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
            ages = [k - born[t] for t in cur]
            label = (
                f"{args.tracker} {args.cam}  frame {k}/{len(images)}  live {len(cur)}  new {new}  lost {len(lost)}"
                f"  median age {np.median(ages) if ages else 0:.0f}"
            )
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if not sink.show(vis):
                break


if __name__ == "__main__":
    main()
