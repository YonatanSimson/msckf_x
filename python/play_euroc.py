#!/usr/bin/env python3
"""Play a EuRoC-format sequence (TUM-VI, EuRoC): all cameras side by side.

Optionally overlays Kalibr AprilGrid detections (--aprilgrid), for checking
the calibration sequences.

    python python/play_euroc.py ~/data/tumvi/dataset-calib-cam1_512_16 --aprilgrid
"""
import argparse
import glob
import os

import cv2
import numpy as np

from viz import FrameSink


def to_u8(im):
    if im.dtype == np.uint16:
        return (im >> 8).astype(np.uint8)
    return im


def aprilgrid_detector():
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2  # Kalibr AprilGrid tags have a 2-bit black border
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(dictionary, params)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="EuRoC-format directory containing mav0/")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--step", type=int, default=1, help="use every N-th frame")
    ap.add_argument("--aprilgrid", action="store_true", help="overlay AprilGrid tag detections")
    ap.add_argument("--tags", type=int, default=36, help="tags on the grid (for the overlay count)")
    ap.add_argument("--out", default="viz_out", help="video output dir when no GUI is available")
    ap.add_argument("--video", action="store_true", help="always write a video instead of a window")
    ap.add_argument("--no-open", action="store_true", help="do not open the written video")
    args = ap.parse_args()

    mav0 = os.path.join(os.path.expanduser(args.sequence), "mav0")
    cams = sorted(d for d in os.listdir(mav0) if d.startswith("cam"))
    files = [sorted(glob.glob(os.path.join(mav0, c, "data", "*.png"))) for c in cams]
    n = min(len(f) for f in files)
    if n == 0:
        raise SystemExit(f"no images under {mav0}/cam*/data")
    print(f"{len(cams)} cameras, {n} frames")

    detector = aprilgrid_detector() if args.aprilgrid else None
    name = os.path.basename(os.path.normpath(os.path.expanduser(args.sequence)))
    with FrameSink(name, args.fps, args.out, args.video, not args.no_open) as sink:
        for i in range(0, n, args.step):
            panels = []
            for c, cam in enumerate(cams):
                im = to_u8(cv2.imread(files[c][i], cv2.IMREAD_UNCHANGED))
                vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
                label = f"{cam}  {i}/{n}"
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(im)
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                    label += f"  tags {0 if ids is None else len(ids)}/{args.tags}"
                cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panels.append(vis)
            gap = np.zeros((panels[0].shape[0], 8, 3), np.uint8)
            row = [p for pair in zip(panels, [gap] * len(panels)) for p in pair][:-1]
            if not sink.show(np.hstack(row)):
                break


if __name__ == "__main__":
    main()
