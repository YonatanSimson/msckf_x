"""Stage 3 gate, simulation: observability of the filter's linearization.

Runs tools/msckf_observability per scenario: one noisy MSCKF run with FEJ and
one without; for windows of 20 images (one every 5 s) it stacks the
observability matrix O of the IMU state and 20 landmarks from the filter's own
Phi (imuTransition) and H (linearizeFeature) at the filter's linearization
points, and checks it against the analytic nullspace N (global translation,
yaw about gravity). Plan, Stage 3 metrics: "numerical rank and |O N| on
simulation, FEJ on vs off; 4 unobservable directions with FEJ".

Gated, with FEJ, every window of every scenario: exactly 4 singular values of
O (columns scaled to unit norm) below 1e-8 of the largest, and
|O n| / (|O| |n|) below 1e-10 for each nullspace direction. Reported: the same
without FEJ, where yaw gains spurious information (3 unobservable directions).
"""

import argparse

import numpy as np

from msckf_gates.common import GateReport, fnum, gate_out, read_csv, run, tool

SCENARIOS = ["handheld", "fast_rotation", "cart"]
UNOBSERVABLE = 4
MAX_ON = 1e-10
DIRECTIONS = ("on_tx", "on_ty", "on_tz", "on_yaw")


def evaluate(report, scenario, rows, fej):
    label = f"{scenario}, {'FEJ' if fej else 'no FEJ'}"
    dims = sorted({int(r["unobservable"]) for r in rows})
    on = max(fnum(r[d]) for r in rows for d in DIRECTIONS)
    on_yaw = np.array([fnum(r["on_yaw"]) for r in rows])
    if fej:
        exact = 100.0 * np.mean([int(r["unobservable"]) == UNOBSERVABLE for r in rows])
        report.check(f"{label}: windows with exactly {UNOBSERVABLE} unobservable directions", exact, ">=", 100.0, "%")
        report.check(f"{label}: max |O n| / (|O| |n|) over windows and directions", on, "<=", MAX_ON)
    else:
        report.note(f"{label}: unobservable directions per window {dims}; yaw |O n| / (|O| |n|) median "
                    f"{np.median(on_yaw):.2g}, max {on_yaw.max():.2g}")  # fmt: skip
    report.note(f"{label}: {len(rows)} windows; 5th smallest singular value / largest, median "
                f"{np.median([fnum(r['sv5']) for r in rows]):.2g}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    out = gate_out("stage3_observability")
    report = GateReport("Stage 3: observability of the linearization (FEJ)")
    for sc in args.scenarios:
        d = out / sc
        run([tool("msckf_observability"), "--scenario", sc, "--duration", args.duration, "--out", d],
            log=out / f"{sc}.log")  # fmt: skip
        rows = read_csv(d / "windows.csv")
        if not rows:
            raise ValueError(f"{d / 'windows.csv'}: no windows")
        for fej in (True, False):
            evaluate(report, sc, [r for r in rows if int(r["fej"]) == int(fej)], fej)
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
