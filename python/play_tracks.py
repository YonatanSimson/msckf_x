#!/usr/bin/env python3
"""Runs a tracker on a recording and plays it back with the tracks.

The tracker is the adopted front end, our vendored Basalt flow with Basalt's
shipped settings (msckf_gates.common.FRONT_END_TUMVI / FRONT_END_EUROC:
frame_to_frame, 40 / 50 px grid, 0.2 px forward-backward, valid-region
mask), unless --tracker klt. Tracks are cached in <out>/<name>/ (--rerun to
recompute), then drawn on every frame:

  dot      live feature with a trail of its last --trail positions
  x        a feature lost in this frame (where it was last)

TUM-VI (a sequence name such as room1 or calib-imu1, or a EuRoC-format
directory; published pinhole-equi calibration): dots are coloured by age,
red when new through yellow to green at --full-age frames.

TartanAir (a trajectory, e.g. AmericanDiner/Data_hard/P000 under
$MSCKF_DATA_LARGE/tartanair, or its path): the run is `tracker_eval
tartanair`, the stage 1 gate's own evaluation, so every observation has its
true pixel from depth and poses. Colours: green within the gate's tolerance,
red for a gross error (as in the gate: error > floor + 2 px, the floor being
the observation's oracle error at ages 1-5 and the sequence's median oracle
error beyond), with a red line to the true pixel (cyan); grey when there is
no truth (depth edge, occluded, out of view). The frames with the most gross
errors are printed, to jump there with --start.

Keys (imshow window): space or a left click pause / resume, n step while paused,
q / Esc quit. The window closes itself at the end of the sequence.

    python python/play_tracks.py room1 [--cam cam1] [--tracker klt] [--start 500]
    python python/play_tracks.py AmericanDiner/Data_hard/P000 [--gross-only]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from msckf_gates.common import (
    FRONT_END_EUROC,
    FRONT_END_TUMVI,
    REPO,
    data_dir,
    euroc_images,
    fnum,
    large_data_dir,
    read_csv,
    run,
    tool,
)
from play_euroc import to_u8
from viz import FrameSink

GROSS_PX = 2.0  # stage1_tracker's TARTAN_GROSS_PX
GREEN, RED, GREY, CYAN, MAGENTA = (0, 200, 0), (0, 0, 255), (150, 150, 150), (255, 255, 0), (255, 0, 255)


@dataclass
class Obs:
    uv: tuple[float, float]
    gt: tuple[float, float] | None = None  # true pixel (TartanAir)
    colour: tuple[int, int, int] | None = None  # None: by age


def age_colour(age: int, full: int) -> tuple[int, int, int]:
    """BGR: red (new) -> yellow -> green (age >= full)."""
    t = min(age / max(full, 1), 1.0)
    return (0, int(255 * min(1.0, 2 * t)), int(255 * min(1.0, 2 * (1 - t))))


def tartan_dir(args: argparse.Namespace) -> Path | None:
    for traj in (Path(args.sequence).expanduser(), args.tartanair / args.sequence):
        if (traj / "image_lcam_front").exists():
            return traj
    return None


def load_tumvi(args: argparse.Namespace) -> tuple[str, list[Path], list[dict[int, Obs]]]:
    root = Path(args.sequence).expanduser()
    if not (root / "mav0").exists():
        root = args.tumvi / f"dataset-{args.sequence}_512_16"
    if not (root / "mav0" / args.cam).exists():
        raise SystemExit(f"{root}/mav0/{args.cam} not found (scripts/data/fetch_tumvi.sh {args.sequence})")
    camchain = args.tumvi / "kalibr_results" / "pinhole-equi-512" / "camchain-imucam-imucalib.yaml"
    name = f"{root.name}_{args.cam}_{args.tracker}"
    d = args.out / name
    if args.rerun or not (d / "tracks.csv").exists():
        extra = FRONT_END_TUMVI if args.tracker == "basalt" else []
        run([tool("tracker_eval"), "tracks", "--euroc", root, "--cam", args.cam, "--camchain", camchain, *extra,
             "--out", d], d.with_suffix(".log"))  # fmt: skip
    images = [p for _, p in euroc_images(root / "mav0" / args.cam)]
    frames: list[dict[int, Obs]] = [{} for _ in images]
    a = np.loadtxt(d / "tracks.csv", delimiter=",", skiprows=1, ndmin=2)
    for k, t, u, v in zip(a[:, 0].astype(int), a[:, 2].astype(np.int64), a[:, 3], a[:, 4]):
        frames[k][int(t)] = Obs((float(u), float(v)))
    return name, images, frames


def load_tartanair(args: argparse.Namespace, traj: Path) -> tuple[str, list[Path], list[dict[int, Obs]]]:
    name = "tartan_" + "_".join(traj.parts[-3:]) + f"_{args.tracker}"
    d = args.out / name
    if args.rerun or not (d / "observations.csv").exists():
        extra = FRONT_END_EUROC if args.tracker == "basalt" else []
        run([tool("tracker_eval"), "tartanair", "--traj", traj, *extra, "--out", d], d.with_suffix(".log"))
    images = sorted((traj / "image_lcam_front").iterdir())
    rows = read_csv(d / "observations.csv")
    oracle = np.array([fnum(r["oracle_px"]) for r in rows])
    floor_median = float(np.nanmedian(oracle)) if np.isfinite(oracle).any() else 0.0
    frames: list[dict[int, Obs]] = [{} for _ in images]
    for r in rows:
        o = Obs((fnum(r["u"]), fnum(r["v"])), colour=GREY)
        err = fnum(r["error_px"])
        if r["status"] == "ok" and np.isfinite(err):
            orc = fnum(r["oracle_px"])
            floor = orc if np.isfinite(orc) else floor_median
            o.gt = (fnum(r["gt_u"]), fnum(r["gt_v"]))
            o.colour = RED if err > floor + GROSS_PX else GREEN
        elif r["age"] == "0" and r["status"] == "ok":
            o.colour = GREEN  # detection: zero error by construction
        frames[int(r["frame"])][int(r["track"])] = o
    gross = [sum(o.colour == RED for o in f.values()) for f in frames]
    worst = np.argsort(gross)[::-1][:10]
    print(f"floor (median oracle error) {floor_median:.2f} px; frames with the most gross errors (--start): "
          + ", ".join(f"{k} ({gross[k]})" for k in worst))  # fmt: skip
    return name, images, frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="TUM-VI name or EuRoC-format directory, or TartanAir trajectory")
    ap.add_argument("--cam", default="cam0", help="TUM-VI camera")
    ap.add_argument("--tracker", choices=["basalt", "klt"], default="basalt")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--tartanair", type=Path, default=large_data_dir() / "tartanair")
    ap.add_argument("--start", type=int, default=0, help="first frame shown")
    ap.add_argument("--fps", type=float, help="playback rate (default: 20 TUM-VI, 10 TartanAir)")
    ap.add_argument("--trail", type=int, default=10, help="trail length, frames")
    ap.add_argument("--full-age", type=int, default=20, help="age drawn fully green, frames (TUM-VI)")
    ap.add_argument("--gross-only", action="store_true", help="TartanAir: draw only gross errors")
    ap.add_argument("--scale", type=float, default=1.5, help="display scale")
    ap.add_argument("--out", type=Path, default=REPO / "tracker_viz_out")
    ap.add_argument("--rerun", action="store_true", help="re-run the tracker even if tracks are cached")
    ap.add_argument("--video", action="store_true", help="write a video instead of opening a window")
    args = ap.parse_args()

    traj = tartan_dir(args)
    name, images, frames = load_tartanair(args, traj) if traj else load_tumvi(args)
    fps = args.fps or (10.0 if traj else 20.0)

    history: dict[int, list[tuple[float, float]]] = {}  # track -> positions, oldest first
    born: dict[int, int] = {}
    prev: dict[int, Obs] = {}
    s = args.scale

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return int(round(p[0] * s)), int(round(p[1] * s))

    with FrameSink(name, fps, str(args.out), args.video) as sink:
        for k, path in enumerate(images):
            cur = frames[k]
            for t, o in cur.items():
                born.setdefault(t, k)
                history.setdefault(t, []).append(o.uv)
                del history[t][: -args.trail]
            lost = {t: o for t, o in prev.items() if t not in cur}
            for t in lost:
                history.pop(t, None)
            new = sum(born[t] == k for t in cur)
            prev = cur
            if k < args.start:
                continue

            img = to_u8(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            for t, o in cur.items():
                col = o.colour or age_colour(k - born[t], args.full_age)
                if args.gross_only and col != RED:
                    continue
                pts = np.round(np.array(history[t]) * s).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
                if col == RED and o.gt is not None:
                    cv2.line(vis, px(o.uv), px(o.gt), RED, 1, cv2.LINE_AA)
                    cv2.circle(vis, px(o.gt), 3, CYAN, 1, cv2.LINE_AA)
                cv2.circle(vis, px(o.uv), 3, col, -1, cv2.LINE_AA)
            if not args.gross_only:
                for o in lost.values():
                    cv2.drawMarker(vis, px(o.uv), MAGENTA, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
            label = f"{args.tracker}  frame {k}/{len(images)}  live {len(cur)}  new {new}  lost {len(lost)}"
            if traj:
                ev = [o for o in cur.values() if o.gt is not None]
                g = sum(o.colour == RED for o in ev)
                label += f"  evaluated {len(ev)}  gross {g} ({100 * g / max(len(ev), 1):.0f}%)"
            else:
                ages = [k - born[t] for t in cur]
                label += f"  median age {np.median(ages) if ages else 0:.0f}"
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if not sink.show(vis):
                break


if __name__ == "__main__":
    main()
