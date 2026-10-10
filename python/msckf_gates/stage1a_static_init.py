"""Stage 1a gate: static initialization.

Simulation (gated): runs the test_static_init unit test, whose Monte Carlo
case requires the mean NEES of tilt (2 dof), gyro bias (3) and tilt + b_g +
b_a (8) inside the 95 % chi-square band over 200 runs; the NEES values it
prints are copied into the report.

EuRoC Machine Hall (gated): runs tools/static_init_eval on MH_01..MH_05 and
requires on every sequence at least one stationary 1 s window with ground
truth, and a tilt error (estimated vs ground-truth gravity direction in the
IMU frame) below 1 deg for every stationary window (plan: gravity direction
error < 1 deg). The MH sequences do not start at rest: the first ~20 s are a
handheld IMU-excitation motion, then the MAV sits on the ground (motors off)
for a few seconds before take-off; that is where the stationary windows are.

Also reported, not gated: TUM-VI room1, which starts handheld, so no window
should pass as stationary (static init is for EuRoC and the cart; TUM-VI tests
dynamic init).

Reported, not gated: tilt NEES on real data, the tilt error with the
ground-truth accelerometer bias as prior (how much of the tilt error is
accelerometer bias), and the gyro bias against EuRoC's ground-truth b_w.
EuRoC's biases come from its own estimator, so they are a reference, not
truth; the report shows how stable our estimate is across windows next to
its offset from that reference.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from msckf_gates.common import (GateReport, build_dir, data_dir, fnum, gate_out, large_data_dir, read_csv,
                                 run, tool)

SEQUENCES = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
MAX_TILT_DEG = 1.0
# Accelerometer bias prior for an uncalibrated ADIS16448-class MEMS IMU (EuRoC):
# zero mean, 0.1 m/s^2 per axis. EuRoC's ground truth puts |b_a| at ~0.15 m/s^2
# (~0.09 per axis); the code default (0.05) assumes a prior calibration.
ACC_BIAS_SIGMA = 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--acc-bias-sigma", type=float, default=ACC_BIAS_SIGMA)
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    args = ap.parse_args()

    out = gate_out("stage1a_static_init")
    report = GateReport("Stage 1a: static initialization")

    # Simulation: the unit test is the gate; its printed NEES go in the report.
    test = build_dir() / "tests" / "test_static_init"
    try:
        log = run([test], log=out / "test_static_init.log")
        sim_ok = 1.0
    except RuntimeError:
        log = (out / "test_static_init.log").read_text()
        sim_ok = 0.0
    report.check("simulation: test_static_init (exact, Monte Carlo NEES in band, rejections)",
                 sim_ok, ">=", 1.0)
    for line in re.findall(r"\[ info \] (mean NEES.*)", log):
        report.note(f"simulation: {line}")

    for seq in SEQUENCES:
        d = args.euroc / seq
        csv_path = out / f"{seq}.csv"
        run([tool("static_init_eval"), "--euroc", d,
                    "--gt", d / "mav0" / "state_groundtruth_estimate0" / "data.csv",
                    "--imu-yaml", d / "mav0" / "imu0" / "sensor.yaml",
                    "--acc-bias-sigma", args.acc_bias_sigma, "--out", csv_path],
                   log=out / f"{seq}.log")
        rows = read_csv(csv_path)
        t0 = fnum(rows[0]["t_end"])
        cmp_rows = [r for r in rows if r["has_gt"] == "1"]
        tilt = np.array([fnum(r["tilt_err_deg"]) for r in cmp_rows])
        report.check(f"{seq}: stationary windows with GT", len(cmp_rows), ">=", 1)
        report.check(f"{seq}: max tilt error", tilt.max() if len(tilt) else np.nan, "<", MAX_TILT_DEG, "deg")

        if not cmp_rows:
            continue
        nees = np.array([fnum(r["tilt_nees"]) for r in cmp_rows])
        oracle = np.array([fnum(r["tilt_err_with_gt_ba_deg"]) for r in cmp_rows])
        bg = np.array([[fnum(r[f"bg_{a}"]) for a in "xyz"] for r in cmp_rows])
        bg_gt = np.array([[fnum(r[f"bg_gt_{a}"]) for a in "xyz"] for r in cmp_rows])
        bg_nees = np.array([fnum(r["bg_nees"]) for r in cmp_rows])
        t_rel = [fnum(r["t_end"]) - t0 for r in cmp_rows]
        report.note(
            f"{seq}: {len(cmp_rows)} stationary windows with GT, window ends at "
            f"{', '.join(f'{t:.1f}' for t in t_rel)} s; "
            f"tilt error mean {tilt.mean():.3f} deg, with GT b_a as prior {np.nanmean(oracle):.3f} deg; "
            f"tilt NEES mean {nees.mean():.2f} (2 dof)")
        report.note(
            f"{seq}: gyro bias spread across windows (std per axis, max) "
            f"{bg.std(axis=0).max() * 1e3:.3f} mrad/s; offset from EuRoC b_w "
            f"{np.linalg.norm((bg - bg_gt).mean(axis=0)) * 1e3:.3f} mrad/s; NEES vs EuRoC b_w mean "
            f"{bg_nees.mean():.1f} (3 dof)")
    room1 = args.tumvi / "dataset-room1_512_16"
    if room1.exists():
        csv_path = out / "tumvi_room1.csv"
        run([tool("static_init_eval"), "--euroc", room1, "--gt", room1 / "dso" / "gt_imu.csv",
             "--imu-yaml", room1 / "dso" / "imu_config.yaml", "--out", csv_path], log=out / "tumvi_room1.log")
        rows = read_csv(csv_path)
        acc = np.array([fnum(r["acc_std"]) for r in rows])
        n_stat = sum(r["stationary"] == "1" for r in rows)
        report.note(f"TUM-VI room1: {n_stat}/{len(rows)} windows stationary; lowest accelerometer std "
                    f"{acc.min():.3f} m/s^2 (threshold 0.1): handheld from the start")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
