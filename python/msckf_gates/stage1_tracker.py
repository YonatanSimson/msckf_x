"""Stage 1 gate: the front-end tracker (plan, "Minimal KLT tracker" and
"Head-to-head against Basalt").

The tracker is the adopted front end, our vendored Basalt flow with Basalt's
shipped settings (msckf_gates.common.FRONT_END_TUMVI / FRONT_END_EUROC);
--tracker klt re-runs the original minimal KLT tracker. Runs
tools/tracker_eval on every evaluation layer and checks, with thresholds
fixed before the full results were known (the plan's targets, not tuned to
either tracker):

Synthetic (gated): the tracker's unit test passes (test_basalt_flow: a known
sub-pixel shift tracked to < 0.05 px RMSE with both flow types, valid-region
mask; test_klt_tracker: the same for KLT, plus LO-MSAC F1 > 0.95 and above
plain RANSAC).

TartanAir (gated, relative to its ground-truth floor): the "oracle" aligns each
detection patch straight into the later frame starting at the true pixel, so
its error is the disagreement of the rendered image with depth and poses.
Over ages 1-5: median tracker error at most 0.1 px above the median oracle
error, and at most 2 % of observations more than 2 px above their oracle.

TUM-VI room1, cam0 and cam1 (gated, relative to a measured floor): residuals
of tracks triangulated from ground-truth poses. The floor comes from AprilGrid
corners (exact static points) on calib-cam1 through the same ground truth.
At most 2 % of observations above 2 px (plan target; the floor's p99 is well
below 2 px), and the median at most 0.3 px above the floor median.

EuRoC MH_01-05, cam0 and cam1 (gated): observation-weighted median track
length >= 8 frames, i.e. the median over all observations of the length of
the track each belongs to (how long the tracks feeding the filter are). The
per-track median is dominated by weak detections that die within a frame or
two, so it is reported with that "infant mortality" instead of gated.
Residual distributions are reported, not gated: there is no floor
measurement in the Machine Hall.

Disparity for initialization (gated): over the 1 s windows that the static-
init gate found IMU-stationary on EuRoC, at least 95 % have median feature
disparity below 1 px (static init's threshold). Moving windows that read as
still are reported. Needs gate_out/stage1a_static_init (run that gate first).

KITTI Flow 2015 (reported, not gated): single-step error against LiDAR flow,
outlier rate (KITTI: > 3 px and > 5 %) and survival. Large driving motion at
10 Hz needs gyro-aided prediction, which is a Stage 2 front-end feature.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates.common import (FRONT_END_EUROC, FRONT_END_TUMVI, GateReport, build_dir, data_dir, fnum, gate_out,
                                large_data_dir, read_csv, run, tool)

EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
CAMS = ["cam0", "cam1"]
TARTAN_SETS = ["RetroOffice/Data_easy", "RetroOffice/Data_hard", "Office/Data_hard", "OldTownFall/Data_easy",
               "OldTownFall/Data_hard", "OldTownNight/Data_hard", "AmericanDiner/Data_hard"]

SYNTH_TEST = {"basalt": "test_basalt_flow", "klt": "test_klt_tracker"}
TARTAN_MAX_EXCESS_PX = 0.1
TARTAN_GROSS_PX = 2.0
TARTAN_MAX_GROSS = 0.02
REAL_OUTLIER_PX = 2.0
REAL_MAX_OUTLIERS = 0.02
REAL_MAX_EXCESS_PX = 0.3
MIN_TRACK_LEN = 8
STATIC_DISPARITY_PX = 1.0
MIN_STATIC_OK = 0.95


def jobs_run(jobs, workers):
    """Runs [(cmd, log)] in parallel; raises if any fails."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(run, cmd, log) for cmd, log in jobs]:
            f.result()


def residuals(path):
    """(rows, residuals of evaluated tracks, per-track lengths, per-observation track lengths)"""
    rows = read_csv(path)
    res = np.array([fnum(r["residual_px"]) for r in rows if r["status"] == "ok"])
    res = res[np.isfinite(res)]
    lens = {}
    for r in rows:
        lens[r["track"]] = int(r["track_len"])
    obs_lens = np.array([int(r["track_len"]) for r in rows])
    return rows, res, np.array(list(lens.values())), obs_lens


def window_disparity(obs_rows, frame_times, t0, t1):
    """Median over features alive from the first to the last frame in [t0, t1]
    of their displacement between those frames."""
    frames = [f for f, t in frame_times.items() if t0 - 1e-6 <= t <= t1 + 1e-6]
    if len(frames) < 2:
        return np.nan
    fa, fb = min(frames), max(frames)
    pos = collections.defaultdict(dict)
    for r in obs_rows:
        f = int(r["frame"])
        if f == fa or f == fb:
            pos[r["track"]][f] = (fnum(r["u"]), fnum(r["v"]))
    d = [np.hypot(p[fb][0] - p[fa][0], p[fb][1] - p[fa][1]) for p in pos.values() if fa in p and fb in p]
    return float(np.median(d)) if d else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--tartanair", type=Path, default=large_data_dir() / "tartanair")
    ap.add_argument("--kitti-flow", type=Path, default=large_data_dir() / "kitti" / "flow2015")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tracker", choices=["basalt", "klt"], default="basalt",
                    help="basalt: the adopted front end (vendored Basalt flow); klt: the original minimal KLT")
    args = ap.parse_args()

    out = gate_out("stage1_tracker")
    names = {"basalt": "vendored Basalt flow", "klt": "minimal KLT"}
    report = GateReport(f"Stage 1: front-end tracker ({names[args.tracker]})")
    te = tool("tracker_eval")
    # Tracker options per dataset (KLT: its defaults).
    tumvi_args = FRONT_END_TUMVI if args.tracker == "basalt" else []
    euroc_args = FRONT_END_EUROC if args.tracker == "basalt" else []
    synth_test = SYNTH_TEST[args.tracker]

    # Synthetic.
    try:
        log = run([build_dir() / "tests" / synth_test], log=out / f"{synth_test}.log")
        ok = 1.0
    except RuntimeError:
        log, ok = (out / f"{synth_test}.log").read_text(), 0.0
    report.check(f"synthetic: {synth_test} passes", ok, ">=", 1.0)
    for line in log.splitlines():
        if "[ info ]" in line:
            report.note("synthetic: " + line.split("[ info ]", 1)[1].strip())

    # Run everything.
    jobs = []
    tum = args.tumvi
    calib = tum / "dataset-calib-cam1_512_16"
    room1 = tum / "dataset-room1_512_16"
    target = tum / "kalibr_results" / "april_6x6_80x80cm.yaml"
    for cam in CAMS:
        jobs.append(([te, "aprilgrid", "--euroc", calib, "--cam", cam, "--gt", calib / "dso" / "gt_imu.csv",
                      "--camchain", calib / "dso" / "camchain.yaml", "--target", target,
                      "--out", out / f"floor_tumvi_{cam}"], out / f"floor_tumvi_{cam}.log"))
        jobs.append(([te, "euroc", "--euroc", room1, "--cam", cam, "--gt", room1 / "dso" / "gt_imu.csv",
                      "--camchain", room1 / "dso" / "camchain.yaml", *tumvi_args, "--out", out / f"tumvi_room1_{cam}"],
                     out / f"tumvi_room1_{cam}.log"))
        for seq in EUROC:
            d = args.euroc / seq
            jobs.append(([te, "euroc", "--euroc", d, "--cam", cam,
                          "--gt", d / "mav0" / "state_groundtruth_estimate0" / "data.csv",
                          *euroc_args, "--out", out / f"euroc_{seq}_{cam}"], out / f"euroc_{seq}_{cam}.log"))
    trajs = sorted(p for s in TARTAN_SETS for p in (args.tartanair / s).glob("P0*") if p.is_dir())
    for p in trajs:
        name = "tartan_" + "_".join(p.parts[-3:])
        jobs.append(([te, "tartanair", "--traj", p, *euroc_args, "--out", out / name], out / f"{name}.log"))
    jobs.append(([te, "kitti-flow", "--kitti", args.kitti_flow, *euroc_args, "--out", out / "kitti_flow"],
                 out / "kitti_flow.log"))
    jobs_run(jobs, args.workers)

    # TartanAir, per set, relative to the oracle floor.
    for s in TARTAN_SETS:
        err, orc, drift = [], [], collections.defaultdict(list)
        for p in trajs:
            if "/".join(p.parts[-3:-1]) != s:
                continue
            for r in read_csv(out / ("tartan_" + "_".join(p.parts[-3:])) / "observations.csv"):
                if r["status"] != "ok":
                    continue
                age, e, o = int(r["age"]), fnum(r["error_px"]), fnum(r["oracle_px"])
                if age in (10, 20):
                    drift[age].append(e)
                if 1 <= age <= 5 and np.isfinite(o):
                    err.append(e)
                    orc.append(o)
        err, orc = np.array(err), np.array(orc)
        excess = float(np.median(err) - np.median(orc))
        gross = float(np.mean(err > orc + TARTAN_GROSS_PX))
        report.check(f"TartanAir {s}: median error above floor (ages 1-5)", excess, "<=", TARTAN_MAX_EXCESS_PX, "px")
        report.check(f"TartanAir {s}: errors > floor + {TARTAN_GROSS_PX:g} px (ages 1-5)", 100 * gross, "<=",
                     100 * TARTAN_MAX_GROSS, "%")
        report.note(f"TartanAir {s}: {len(err)} obs (ages 1-5); tracker median {np.median(err):.3f} px, floor "
                    f"median {np.median(orc):.3f} px; drift: age 10 median "
                    f"{np.median(drift[10]) if drift[10] else float('nan'):.2f} px, age 20 median "
                    f"{np.median(drift[20]) if drift[20] else float('nan'):.2f} px")

    # TUM-VI room1 against the AprilGrid floor.
    for cam in CAMS:
        _, floor, _, _ = residuals(out / f"floor_tumvi_{cam}" / "observations.csv")
        _, res, lens, obs_lens = residuals(out / f"tumvi_room1_{cam}" / "observations.csv")
        outliers = float(np.mean(res > REAL_OUTLIER_PX))
        report.check(f"TUM-VI room1 {cam}: residuals > {REAL_OUTLIER_PX:g} px", 100 * outliers, "<",
                     100 * REAL_MAX_OUTLIERS, "%")
        report.check(f"TUM-VI room1 {cam}: median residual above AprilGrid floor",
                     float(np.median(res) - np.median(floor)), "<=", REAL_MAX_EXCESS_PX, "px")
        report.note(f"TUM-VI {cam}: floor (AprilGrid, calib-cam1) median {np.median(floor):.3f}, p90 "
                    f"{np.percentile(floor, 90):.3f}, p99 {np.percentile(floor, 99):.3f} px; room1 tracks median "
                    f"{np.median(res):.3f}, p90 {np.percentile(res, 90):.3f}, p99 {np.percentile(res, 99):.3f} px; "
                    f"track length: observation-weighted median {np.median(obs_lens):.0f}, per-track median "
                    f"{np.median(lens):.0f}, tracks dying within 2 frames {100 * np.mean(lens <= 2):.0f}%")

    # EuRoC: track length gated, residuals reported.
    for seq in EUROC:
        for cam in CAMS:
            d = out / f"euroc_{seq}_{cam}"
            _, res, lens, obs_lens = residuals(d / "observations.csv")
            fr = read_csv(d / "frames.csv")
            feats = np.array([fnum(r["features"]) for r in fr])
            ms = np.array([fnum(r["time_ms"]) for r in fr])
            report.check(f"EuRoC {seq} {cam}: observation-weighted median track length", float(np.median(obs_lens)),
                         ">=", MIN_TRACK_LEN, "frames")
            report.note(f"EuRoC {seq} {cam}: residual median {np.median(res):.3f}, p90 "
                        f"{np.percentile(res, 90):.3f} px, > {REAL_OUTLIER_PX:g} px "
                        f"{100 * np.mean(res > REAL_OUTLIER_PX):.1f}%; "
                        f"features/frame median {np.median(feats):.0f}; {np.median(ms):.1f} ms/frame (median); "
                        f"per-track median length {np.median(lens):.0f}, mean {lens.mean():.1f}, "
                        f"tracks dying within 2 frames {100 * np.mean(lens <= 2):.0f}%")

    # Disparity in the static-init windows.
    s1a = gate_out("stage1a_static_init")
    static_ok = static_n = moving_still = moving_n = 0
    for seq in EUROC:
        wpath = s1a / f"{seq}.csv"
        if not wpath.exists():
            continue
        d = out / f"euroc_{seq}_cam0"
        obs = read_csv(d / "observations.csv")
        ftimes = {int(r["frame"]): fnum(r["t"]) for r in read_csv(d / "frames.csv")}
        for w in read_csv(wpath):
            t1 = fnum(w["t_end"])
            disp = window_disparity(obs, ftimes, t1 - 1.0, t1)
            if not np.isfinite(disp):
                continue
            if w["stationary"] == "1":
                static_n += 1
                static_ok += disp < STATIC_DISPARITY_PX
            else:
                moving_n += 1
                moving_still += disp < STATIC_DISPARITY_PX
    report.check("EuRoC cam0: IMU-stationary windows with disparity < 1 px",
                 100 * static_ok / static_n if static_n else np.nan, ">=", 100 * MIN_STATIC_OK, "%")
    report.note(f"disparity: {static_ok}/{static_n} stationary windows below {STATIC_DISPARITY_PX:g} px; "
                f"{moving_still}/{moving_n} moving windows also below it (the IMU test catches those)")

    # KITTI flow, reported.
    rows = read_csv(out / "kitti_flow" / "observations.csv")
    st = collections.Counter(r["status"] for r in rows)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    e = np.array([fnum(r["error_px"]) for r in ok_rows])
    fl = np.array([fnum(r["flow_px"]) for r in ok_rows])
    tracked = st["ok"] + st["no_gt"]
    report.note(f"KITTI flow 2015: {len(e)} tracked features with ground truth over 200 pairs; error median "
                f"{np.median(e):.3f} px, mean {e.mean():.3f} px; KITTI outliers (> 3 px and > 5 %) "
                f"{100 * np.mean((e > 3) & (e > 0.05 * fl)):.1f}%; "
                f"survival {100 * tracked / (tracked + st['lost']):.0f}% "
                f"of detections")
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 1000)]:
        sel = (fl >= lo) & (fl < hi)
        if sel.any():
            report.note(f"KITTI flow {lo}-{hi} px: n {sel.sum()}, median error {np.median(e[sel]):.3f} px, "
                        f"> 3 px {100 * np.mean(e[sel] > 3):.1f}%")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
