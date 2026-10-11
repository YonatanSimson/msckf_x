#!/usr/bin/env python3
"""Tunes the real-data noise inflation of dynamic initialization (plan,
Stage 1b): IMU noise densities scaled by --imu-noise-scale, bearing noise of
--pixel-sigma px, on sequences that no gate uses, so the gate does not judge
settings tuned on itself.

  EuRoC V1_01-03 (Vicon Room 1), cam0, EuRoC's ground-truth biases
  TUM-VI room2, cam0, zero bias (its IMU data is calibrated)

For every (scale, pixel sigma) of the grid, dynamic_init_eval runs on each
sequence (windows 0.5 and 1 s, plus the retry pipeline) with tracks from the
adopted front end, and the velocity NEES at the hand-off (3 dof, yaw free) is
collected over accepted solutions. Reported per setting: mean NEES with its
95 % band, share above chi2_0.99(3), success rate and velocity error median.
The recommendation is the least inflation (scale first, then pixel sigma)
whose mean NEES is inside the band.

    python python/tune_init_noise.py [--scales 1,2,4,8] [--pixels 1,1.5,2,3]

Writes tune_init_noise_out/report.md and metrics.json. Data:
scripts/data/fetch_euroc.sh vicon_room1, scripts/data/fetch_tumvi.sh room2.
"""

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates.common import (
    FRONT_END_EUROC,
    FRONT_END_TUMVI,
    REPO,
    data_dir,
    fnum,
    large_data_dir,
    nees_band,
    read_csv,
    run,
    tool,
    write_metrics,
)

CHI2_99_3DOF = 11.345


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--scales", default="1,2,4,8")
    ap.add_argument("--pixels", default="1,1.5,2,3")
    ap.add_argument("--out", type=Path, default=REPO / "tune_init_noise_out")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    te, de = tool("tracker_eval"), tool("dynamic_init_eval")
    seqs = {}
    for s in ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]:
        d = args.euroc / s
        seqs[f"euroc_{s}"] = (["--euroc", d, "--cam", "cam0"], FRONT_END_EUROC,
                              ["--gt", d / "mav0" / "state_groundtruth_estimate0" / "data.csv", "--bias", "gt",
                               "--imu-yaml", d / "mav0" / "imu0" / "sensor.yaml"])  # fmt: skip
    room2 = args.tumvi / "dataset-room2_512_16"
    seqs["tumvi_room2"] = (["--euroc", room2, "--cam", "cam0", "--camchain", room2 / "dso" / "camchain.yaml"],
                           FRONT_END_TUMVI,
                           ["--gt", room2 / "dso" / "gt_imu.csv", "--bias", "zero", "--imu-yaml",
                            args.tumvi / "kalibr_results" / "pinhole-equi-512" / "imu-imucalib.yaml"])  # fmt: skip
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in [
            ex.submit(run, [te, "tracks", *cam, *fe, "--out", out / f"tracks_{n}"], out / f"tracks_{n}.log")
            for n, (cam, fe, _) in seqs.items()
            if not (out / f"tracks_{n}" / "tracks.csv").exists()
        ]:
            f.result()  # fmt: skip
        grid = list(itertools.product(args.scales.split(","), args.pixels.split(",")))
        jobs = []
        for sc, px in grid:
            for n, (cam, _, rest) in seqs.items():
                d = out / f"s{sc}_p{px}" / n
                cmd = [de, "euroc", *cam, *rest, "--tracks", out / f"tracks_{n}" / "tracks.csv", "--windows",
                       "0.5,1.0", "--imu-noise-scale", sc, "--pixel-sigma", px, "--out", d]  # fmt: skip
                jobs.append(ex.submit(run, cmd, d.with_suffix(".log")))
        for f in jobs:
            f.result()

    lines = ["# Dynamic-init noise inflation (tuning sequences: EuRoC V1_01-03, TUM-VI room2)", "",
             "| IMU noise scale | Pixel sigma | Solutions | Velocity NEES mean (band) | > chi2_0.99 | Success | "
             "Velocity error median |", "| --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    metrics, best = [], None
    for sc, px in grid:
        rows = [r for n in seqs for r in read_csv(out / f"s{sc}_p{px}" / n / "runs.csv")]
        sol = [r for r in rows if r["method"] in ("refined", "retry")]
        acc = [r for r in sol if r["ok"] == "1" and np.isfinite(fnum(r["velocity_nees"]))]
        nees = np.array([fnum(r["velocity_nees"]) for r in acc])
        verr = np.array([fnum(r["velocity_err"]) for r in acc])
        lo, hi = nees_band(len(nees), 3) if len(nees) else (np.nan, np.nan)
        mean = float(nees.mean()) if len(nees) else np.nan
        success = len(acc) / max(len(sol), 1)
        lines.append(f"| {sc} | {px} | {len(nees)} | {mean:.2f} ({lo:.2f}-{hi:.2f}) | "
                     f"{100 * np.mean(nees > CHI2_99_3DOF) if len(nees) else np.nan:.0f} % | {100 * success:.0f} % | "
                     f"{np.median(verr) if len(verr) else np.nan:.3f} m/s |")  # fmt: skip
        metrics.append({"name": f"s{sc}_p{px}/velocity_nees_mean", "value": mean, "better": None})
        if best is None and lo <= mean <= hi:
            best = (sc, px)
    lines += ["", "Recommended (least inflation with the mean NEES in band): "
              + (f"IMU noise x{best[0]}, {best[1]} px" if best else "none in the grid")]  # fmt: skip
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_metrics(out, metrics)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
