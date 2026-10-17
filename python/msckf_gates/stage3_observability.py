"""Stage 3 gate, simulation: observability of the filter's linearization.

Runs tools/msckf_observability per scenario: one noisy MSCKF run with FEJ and
one without; for windows of 20 images (one every 5 s) it stacks the
observability matrix O of the IMU state and 20 landmarks from the transitions
the filter applied to P and H (linearizeFeature) at the clones' first
estimates (FEJ) or updated estimates (no FEJ, approximate), and checks it
against the analytic nullspace N (global translation, yaw about gravity).
Plan, Stage 3 metrics: "numerical rank and |O N| on simulation, FEJ on vs off;
4 unobservable directions with FEJ".

Gated per scenario, with FEJ: at least 4 windows, every window with exactly 4
unobservable directions (nullity of O, columns scaled to unit norm, singular
values below 1e-8 of the largest) and |O n| / (|O|_F |n|) below 1e-10 for each
nullspace direction. Control, without FEJ: yaw gains spurious information, so
the median yaw |O n| / (|O|_F |n|) must exceed 1e-6 (else the comparison is
broken and the FEJ result proves nothing).
"""

import argparse
from pathlib import Path

import numpy as np

from msckf_gates.common import GateReport, fnum, gate_out, read_csv, run, tool

SCENARIOS: list[str] = ["handheld", "fast_rotation", "cart"]
UNOBSERVABLE = 4
MAX_ON = 1e-10
MIN_YAW_ON_NO_FEJ = 1e-6
MIN_WINDOWS = 4
DIRECTIONS = ("on_tx", "on_ty", "on_tz", "on_yaw")


def check_fej(report: GateReport, label: str, rows: list[dict[str, str]]) -> None:
    exact: float = 100.0 * float(np.mean([int(r["unobservable"]) == UNOBSERVABLE for r in rows]))
    on: float = max(fnum(r[d]) for r in rows for d in DIRECTIONS)
    report.check(f"{label}: windows", len(rows), ">=", MIN_WINDOWS)
    report.check(f"{label}: windows with exactly {UNOBSERVABLE} unobservable directions", exact, ">=", 100.0, "%")
    report.check(f"{label}: max |O n| / (|O|_F |n|) over windows and directions", on, "<=", MAX_ON)


def check_no_fej(report: GateReport, label: str, rows: list[dict[str, str]]) -> None:
    on_yaw: np.ndarray = np.array([fnum(r["on_yaw"]) for r in rows])
    dims: list[int] = sorted({int(r["unobservable"]) for r in rows})
    report.check(f"{label} (control): median yaw |O n| / (|O|_F |n|)", float(np.median(on_yaw)), ">=",
                 MIN_YAW_ON_NO_FEJ)  # fmt: skip
    report.note(f"{label}: unobservable directions per window {dims}; yaw |O n| / (|O|_F |n|) max {on_yaw.max():.2g}")


def evaluate(report: GateReport, scenario: str, rows: list[dict[str, str]], fej: bool) -> None:
    label = f"{scenario}, {'FEJ' if fej else 'no FEJ'}"
    if not rows:
        raise ValueError(f"{label}: no windows (see {scenario}.log)")
    (check_fej if fej else check_no_fej)(report, label, rows)
    sv5: float = float(np.median([fnum(r["sv5"]) for r in rows]))
    report.note(f"{label}: {len(rows)} windows; 5th smallest singular value / largest, median {sv5:.2g}")


def scenario_rows(out: Path, scenario: str, duration: float) -> list[dict[str, str]]:
    d = out / scenario
    run([tool("msckf_observability"), "--scenario", scenario, "--duration", duration, "--out", d],
        log=out / f"{scenario}.log")  # fmt: skip
    return read_csv(d / "windows.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    out: Path = gate_out("stage3_observability")
    report = GateReport("Stage 3: observability of the linearization (FEJ)")
    for sc in args.scenarios:
        rows = scenario_rows(out, sc, args.duration)
        for fej in (True, False):
            evaluate(report, sc, [r for r in rows if int(r["fej"]) == int(fej)], fej)
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
