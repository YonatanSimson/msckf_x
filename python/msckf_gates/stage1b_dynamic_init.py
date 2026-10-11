"""Stage 1b gate: dynamic initialization (plan, "Dynamic initialization" and
"Stage 1 metrics").

Runs tools/dynamic_init_eval with the paper's protocol (an initialization at
the start of every 10 s segment, windows 0.1-1.0 s) on:

  simulation   handheld, car (KITTI-like), cart and fast-rotation scenarios,
               10 seeds each,
               biases known
  EuRoC        MH_01-05, cam0 (left camera only, as in the paper), biases from
               EuRoC's ground truth as the "prior calibration" the paper assumes
  TUM-VI       room1, cam0; zero bias (its IMU data is calibrated)
  KITTI raw    every drive converted by python/kitti_raw_to_euroc.py (OXTS
               IMU at 100 Hz, zero bias, a start every 5 s); reported only

with bearings from the adopted front end (tracker_eval tracks with
msckf_gates.common.FRONT_END_*) and each dataset's IMU noise densities. Three
methods per window (see dynamic_init_eval): linear (step 1), refined (step 1
+ step 2, the pipeline) and okvis (OKVIS2-X's start: gravity from the mean
accelerometer reading, velocity 0; a baseline).

Gated, per dataset (simulated handheld, EuRoC MH_01-05, TUM-VI room1), for
the pipeline with retry (initializeDynamic: from each start the window grows
from 0.5 s by 0.25 s up to 5 s until a solution passes), as the plan's Stage 1
metrics define them:
  success rate >= 95 %: initialized within 5 s (the plan's conditions 1 and
      2, the linear system solves and the refinement converges with a valid
      covariance passing the scale gate; condition 3, VIO ATE over the next
      10 s, needs Stage 3)
  time to initialize, median <= 1 s (plan: <= 1 s handheld)
  gravity direction error, median < 1 deg
  velocity error, median < 0.1 m/s
  simulation only: mean velocity NEES at the hand-off (3 dof, yaw free) in its
      95 % band (plan: initial NEES on simulation)
Car, cart, fast rotation and KITTI are reported. The plan's scale metric is a
Sim(3) fit over 10 s of VIO (Stage 3); each solution's window scale is
reported. Every method at fixed window lengths (the paper's protocol) is
reported as diagnostics.
"""

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates.common import (
    FRONT_END_EUROC,
    FRONT_END_TUMVI,
    GateReport,
    data_dir,
    fnum,
    gate_out,
    large_data_dir,
    nees_band,
    read_csv,
    run,
    tool,
)

EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
SCENARIOS = ["handheld", "car", "cart", "fast_rotation"]
# Real-data noise inflation, tuned by python/tune_init_noise.py on EuRoC
# V1_01-03 and TUM-VI room2 (no gate uses them): IMU noise densities x4 and
# 2 px bearings bring V1_02, V1_03 and room2 into their NEES bands (V1_01, slow
# motion, stays ~3x overconfident: open). Simulation is not inflated.
REAL_NOISE = ["--imu-noise-scale", "4", "--pixel-sigma", "2"]
METHODS = ["refined", "linear", "okvis"]  # per window length; "retry" is summarized separately
GATED = ("sim handheld", "EuRoC MH_01-05", "TUM-VI room1")
MAX_WINDOW = 5.0  # s, the retry limit (DynamicInitConfig::max_window)
MIN_SUCCESS = 0.95
MAX_TIME_TO_INIT = 1.0  # s, median (plan: <= 1 s handheld)
MAX_GRAVITY_DEG = 1.0
MAX_VELOCITY = 0.1  # m/s, or MAX_VELOCITY_REL of the speed
MAX_VELOCITY_REL = 0.10
MAX_SCALE_PCT = 5.0


def accurate(r):
    if r["ok"] != "1":
        return False
    v_err, v = fnum(r["velocity_err"]), fnum(r["velocity_norm"])
    scale = fnum(r["scale_err_pct"])
    return (
        fnum(r["gravity_err_deg"]) < MAX_GRAVITY_DEG
        and (v_err < MAX_VELOCITY or v_err < MAX_VELOCITY_REL * v)
        and (r["method"] == "okvis" or scale < MAX_SCALE_PCT)
    )


def summarize(rows):
    """Per window length: dict of stats."""
    out = {}
    for w in sorted({fnum(r["window"]) for r in rows}):
        sel = [r for r in rows if fnum(r["window"]) == w]
        acc = [r for r in sel if r["ok"] == "1"]

        def med(key, rs=acc):
            x = np.array([fnum(r[key]) for r in rs])
            x = x[np.isfinite(x)]
            return float(np.median(x)) if len(x) else float("nan")

        out[w] = {
            "runs": len(sel),
            "accepted": len(acc) / max(len(sel), 1),
            "accurate": sum(accurate(r) for r in sel) / max(len(sel), 1),
            "gravity": med("gravity_err_deg"),
            "velocity": med("velocity_err"),
            "scale": med("scale_err_pct"),
            "ms": med("time_ms", sel),
            "reasons": collections.Counter(r["reason"] for r in sel if r["ok"] != "1"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument(
        "--kitti",
        type=Path,
        default=large_data_dir() / "kitti" / "euroc",
        help="KITTI raw drives in EuRoC layout (python/kitti_raw_to_euroc.py); reported if present",
    )
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = gate_out("stage1b_dynamic_init")
    report = GateReport("Stage 1b: dynamic initialization (pipeline with retry)")
    te, de = tool("tracker_eval"), tool("dynamic_init_eval")

    # Front-end tracks, then the initializer runs.
    room1 = args.tumvi / "dataset-room1_512_16"
    tum_cam = ["--euroc", room1, "--cam", "cam0", "--camchain", room1 / "dso" / "camchain.yaml"]
    real = {f"euroc_{s}": (["--euroc", args.euroc / s, "--cam", "cam0"], FRONT_END_EUROC,
                           args.euroc / s / "mav0" / "state_groundtruth_estimate0" / "data.csv", "gt",
                           args.euroc / s / "mav0" / "imu0" / "sensor.yaml")
            for s in EUROC}  # fmt: skip
    real["tumvi_room1"] = (
        tum_cam,
        FRONT_END_TUMVI,
        room1 / "dso" / "gt_imu.csv",
        "zero",
        args.tumvi / "kalibr_results" / "pinhole-equi-512" / "imu-imucalib.yaml",
    )
    # KITTI raw drives (converted by python/kitti_raw_to_euroc.py): reported,
    # not gated; short drives, so a start every 5 s.
    extra = {}
    kitti = sorted(p for p in args.kitti.glob("*_drive_*") if (p / "mav0").exists())
    for d in kitti:
        name = f"kitti_{d.name}"
        real[name] = (["--euroc", d, "--cam", "cam0"], FRONT_END_EUROC, d / "gt_imu.csv", "zero",
                      d / "mav0" / "imu0" / "sensor.yaml")  # fmt: skip
        # The converter writes the OXTS noise measured against ground truth
        # (python/kitti_imu_consistency.py): no further IMU inflation.
        extra[name] = ["--segment", "5", "--imu-noise-scale", "1"]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(run, [te, "tracks", *cam, *fe, "--out", out / f"tracks_{name}"], out / f"tracks_{name}.log")
            for name, (cam, fe, _, _, _) in real.items()
            if not (out / f"tracks_{name}" / "tracks.csv").exists()
        ]
        futures += [ex.submit(run, [de, "sim", "--scenario", s, "--seeds", args.seeds, "--out", out / f"sim_{s}"],
                              out / f"sim_{s}.log") for s in SCENARIOS]  # fmt: skip
        for f in futures:
            f.result()
        futures = [ex.submit(run, [de, "euroc", *cam, "--gt", gt, "--tracks", out / f"tracks_{name}" / "tracks.csv",
                                   "--imu-yaml", imu, "--bias", bias, *REAL_NOISE, *extra.get(name, []),
                                   "--out", out / name],
                             out / f"{name}.log")
                   for name, (cam, _, gt, bias, imu) in real.items()]  # fmt: skip
        for f in futures:
            f.result()

    datasets = {f"sim {s}": read_csv(out / f"sim_{s}" / "runs.csv") for s in SCENARIOS}
    datasets["EuRoC MH_01-05"] = [r for s in EUROC for r in read_csv(out / f"euroc_{s}" / "runs.csv")]
    datasets["TUM-VI room1"] = read_csv(out / "tumvi_room1" / "runs.csv")
    if kitti:
        datasets[f"KITTI raw ({len(kitti)} drives)"] = [
            r for d in kitti for r in read_csv(out / f"kitti_{d.name}" / "runs.csv")
        ]
    # Gated: the pipeline with retry (initializeDynamic), as the system runs.
    for name, rows in datasets.items():
        rr = [r for r in rows if r["method"] == "retry"]
        if not rr:
            continue
        ok = [r for r in rr if r["ok"] == "1"]
        t = np.array([fnum(r["window"]) for r in ok])
        grav = np.array([fnum(r["gravity_err_deg"]) for r in ok])
        vel = np.array([fnum(r["velocity_err"]) for r in ok])
        nees = np.array([fnum(r["velocity_nees"]) for r in ok])
        nees = nees[np.isfinite(nees)]
        nan = float("nan")
        success = len(ok) / len(rr)
        t_med = float(np.median(t)) if len(t) else nan
        if name in GATED:
            report.check(
                f"{name}: success rate (retry up to {MAX_WINDOW:g} s)", 100 * success, ">=", 100 * MIN_SUCCESS, "%"
            )
            report.check(f"{name}: time to initialize, median", t_med, "<=", MAX_TIME_TO_INIT, "s")
            report.check(
                f"{name}: gravity error median",
                float(np.median(grav)) if len(grav) else nan,
                "<",
                MAX_GRAVITY_DEG,
                "deg",
            )
            report.check(f"{name}: velocity error median", float(np.median(vel)) if len(vel) else nan, "<",
                         MAX_VELOCITY, "m/s")  # fmt: skip
            if name.startswith("sim"):
                lo, hi = nees_band(len(nees), 3) if len(nees) else (nan, nan)
                mean = float(nees.mean()) if len(nees) else nan
                report.check(f"{name}: velocity NEES mean <= band upper {hi:.2f} (3 dof)", mean, "<=", hi)
                report.check(f"{name}: velocity NEES mean >= band lower {lo:.2f} (3 dof)", mean, ">=", lo)
        fails = collections.Counter(r["reason"] for r in rr if r["ok"] != "1")
        report.note(f"{name}, retry: success {len(ok)}/{len(rr)} ({100 * success:.0f}%), time to initialize median "
                    f"{t_med:.2f} s, p90 {np.percentile(t, 90) if len(t) else nan:.2f} s; accepted: gravity "
                    f"{np.median(grav) if len(grav) else nan:.2f} deg, velocity "
                    f"{np.median(vel) if len(vel) else nan:.3f} m/s (medians), velocity NEES mean "
                    f"{nees.mean() if len(nees) else nan:.2f} (3 dof); failures: "
                    + (", ".join(f"{k} {v}" for k, v in fails.most_common()) or "-"))  # fmt: skip

    # Reported: each method at fixed window lengths (diagnostics; the
    # protocol of the paper's tables).
    for name, rows in datasets.items():
        for method in METHODS:
            for w, st in summarize([r for r in rows if r["method"] == method]).items():
                reasons = ", ".join(f"{k} {v}" for k, v in st["reasons"].most_common()) or "-"
                report.note(f"{name}, {method}, {w:g} s: {st['runs']} runs, accepted {100 * st['accepted']:.0f}%, "
                            f"accurate {100 * st['accurate']:.0f}%; accepted runs' median errors: gravity "
                            f"{st['gravity']:.2f} deg, velocity {st['velocity']:.3f} m/s, scale {st['scale']:.1f}%; "
                            f"rejected: {reasons}; {st['ms']:.2f} ms")  # fmt: skip
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
