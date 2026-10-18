"""Stage 3 gate, simulation: MSCKF consistency and calibration convergence.

Runs tools/msckf_sim_eval (Monte Carlo over IMU noise, pixel noise and the
initial error; every run shares the trajectory) per scenario, with FEJ (gated)
and without FEJ (reported, to measure what FEJ buys). Gated scenarios match the
priority datasets (user, 2026-10-11): handheld (TUM-VI, EuRoC) and
fast_rotation (UZH-FPV). The cart is reported only (known overconfident at
1 px, see the plan's Stage 3 status). Plan, Stage 3 metrics:

- Consistency (gated, FEJ): the Monte Carlo average NEES of orientation and of
  position (3 dof each) inside the chi-square 95 % band for the number of runs
  in at least MIN_INSIDE of the epochs. The band is for one epoch's average;
  the run-long mean averages far more (correlated) samples, so checking it
  against the same band would pass a filter that is off at every epoch.
- Calibration convergence (gated): RMSE over runs of the final extrinsic
  rotation / translation and time-offset errors, < 0.1 deg, < 1 mm, < 0.5 ms.
- Reported: run-long mean NEES, final position error RMSE and
  drift (% of path length), per-image filter time.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates.common import GateReport, fnum, gate_out, nees_band, read_csv, run, tool

GATED = ["handheld", "fast_rotation"]
REPORTED = ["cart"]
MAX_EXT_ROT_DEG = 0.1
MAX_EXT_POS_M = 1e-3
MAX_TD_MS = 0.5
MIN_RUN_SHARE = 0.99  # shortest run / longest
MIN_INSIDE = 90.0  # % of epochs; 95 % for independent epochs, some slack for correlation


def load(path, expected_runs):
    """Every run shares the trajectory, so all cover it to within a frame or
    two; one that stopped early would cut every run short."""
    rows = read_csv(path)
    runs = sorted({int(r["run"]) for r in rows})
    by_run = {k: [r for r in rows if int(r["run"]) == k] for k in runs}
    lengths = [len(v) for v in by_run.values()]
    n = min(lengths, default=0)
    if len(runs) != expected_runs or n < MIN_RUN_SHARE * max(lengths):
        raise ValueError(f"{path}: {len(runs)} of {expected_runs} runs, epochs per run {n}-{max(lengths, default=0)}")

    def col(name):  # runs x epochs
        return np.array([[fnum(r[name]) for r in by_run[k][:n]] for k in runs])

    return col, len(runs)


def evaluate(report, label, col, runs, gated):
    lo, hi = nees_band(runs, 3)
    for kind in ("ori", "pos"):
        avg = col(f"nees_{kind}").mean(axis=0)  # Monte Carlo average per epoch
        inside = np.mean((avg >= lo) & (avg <= hi)) * 100
        name = f"{label}: mean {'orientation' if kind == 'ori' else 'position'} NEES (3 dof, {runs} runs)"
        if gated:
            report.check(name + f", epochs inside band [{lo:.2f}, {hi:.2f}]", inside, ">=", MIN_INSIDE, "%")
        else:
            report.note(f"{name}: epochs inside band [{lo:.2f}, {hi:.2f}] {inside:.0f} %")
        report.note(f"{name}: run-long mean {avg.mean():.2f}")
    pos = col("pos_err_m")[:, -1]
    rot = col("rot_err_deg")[:, -1]
    report.note(
        f"{label}: final error RMSE over runs: position {np.sqrt(np.mean(pos**2)):.3f} m, "
        f"rotation {np.sqrt(np.mean(rot**2)):.3f} deg; per-image time median "
        f"{np.median(col('time_ms')):.2f} ms, p99 {np.percentile(col('time_ms'), 99):.2f} ms"
    )


def variant_cmd(args: argparse.Namespace, variant: tuple[str, bool], d: Path) -> list:
    """msckf_sim_eval command line for one scenario and FEJ setting."""
    sc, fej = variant
    cmd = [tool("msckf_sim_eval"), "--scenario", sc, "--runs", args.runs, "--duration", args.duration, "--out", d]
    if not fej:
        cmd.append("--no-fej")
    if args.min_parallax is not None:
        cmd += ["--min-parallax", args.min_parallax]
    if args.clones is not None:
        cmd += ["--clones", args.clones]
    return cmd


def check_calibration(report: GateReport, sc: str, col: Callable[[str], np.ndarray]) -> None:
    ext_rot, ext_pos, td = (col(c)[:, -1] for c in ("ext_rot_err_deg", "ext_pos_err_m", "td_err_ms"))
    rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
    report.check(f"{sc}: final extrinsic rotation error RMSE", rms(ext_rot), "<", MAX_EXT_ROT_DEG, "deg")
    report.check(f"{sc}: final extrinsic translation error RMSE", rms(ext_pos), "<", MAX_EXT_POS_M, "m")
    report.check(f"{sc}: final time-offset error RMSE", rms(td), "<", MAX_TD_MS, "ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--scenarios", nargs="+", default=GATED)
    ap.add_argument("--reported", nargs="*", default=REPORTED)
    ap.add_argument("--min-parallax", type=float, help="deg, overrides the filter's parallax gate (diagnosis)")
    ap.add_argument("--clones", type=int, help="overrides the filter's window size max_clones (diagnosis)")
    ap.add_argument("--workers", type=int, default=6, help="variants run in parallel")
    ap.add_argument("--out", default="stage3_sim", help="gate_out subdirectory")
    args = ap.parse_args()

    out = gate_out(args.out)
    report = GateReport("Stage 3: MSCKF on simulation")
    scenarios = list(dict.fromkeys(args.scenarios + args.reported))  # a scenario in both lists runs once
    variants = [(sc, fej) for sc in scenarios for fej in (True, False)]
    tag = {v: f"{v[0]}_{'fej' if v[1] else 'nofej'}" for v in variants}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = [ex.submit(run, variant_cmd(args, (sc, fej), out / tag[sc, fej]), out / f"{tag[sc, fej]}.log")
                for sc, fej in variants]  # fmt: skip
        for j in jobs:
            j.result()
    for sc, fej in variants:
        gated_sc = sc in args.scenarios
        col, runs = load(out / tag[sc, fej] / "epochs.csv", args.runs)
        evaluate(report, f"{sc}, {'FEJ' if fej else 'no FEJ'}", col, runs, gated=fej and gated_sc)
        if fej and gated_sc:
            check_calibration(report, sc, col)
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
