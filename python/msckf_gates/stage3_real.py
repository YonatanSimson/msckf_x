"""Stage 3 gate, real data: the MSCKF on recorded sequences (plan, Stage 3
metrics: accuracy, robustness, runtime).

Runs tools/msckf_run on EuRoC MH_01-05, TUM-VI room1 and LaMAria R_01-03 (if
converted) with the Stage 1b front-end tracks (msckf_gates.sequences, cached
by the Stage 1b gate), the Stage 1b initialization and the same noise
inflation (IMU x4, 2 px), then scores each trajectory with tools/msckf_eval
(ATE after a position + yaw alignment, relative error).

Gated (plan, "Robustness: 0 divergences on all test sequences"): every
sequence initializes and the filter never diverges: no non-finite or runaway
state (msckf_run) and ATE RMSE at most 10 % of the distance travelled (a
filter that has lost track can stay finite: EuRoC MH_03 at 30 clones reached
586 m ATE over 131 m). Recorded for
benchmark_compare, not gated until the OpenVINS mono baseline exists (plan:
"ATE within 20 % of OpenVINS mono, run as a binary"): ATE RMSE, rotation ATE
RMSE, mean relative translation error and the back end's time per image.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates import sequences
from msckf_gates.common import GateReport, data_dir, fnum, gate_out, large_data_dir, read_csv, run, tool
from msckf_gates.sequences import Sequence


def real_sequences(args: argparse.Namespace, report: GateReport) -> list[Sequence]:
    lamaria = [args.lamaria / s for s in sequences.LAMARIA]
    missing = [d.name for d in lamaria if not (d / "gt_imu.csv").exists()]
    if missing:
        report.note(f"LaMAria not run, missing {', '.join(missing)}")
        lamaria = []
    return [*sequences.euroc(args.euroc), sequences.tumvi_room1(args.tumvi), *sequences.converted("lamaria", lamaria)]


def run_cmd(seq: Sequence, args: argparse.Namespace, d: Path) -> list:
    cmd = [tool("msckf_run"), *seq.cam, "--tracks", sequences.tracks_csv(seq), "--imu-yaml", seq.imu_yaml,
           "--gt", seq.gt, "--bias", seq.bias, *sequences.REAL_NOISE, "--out", d]  # fmt: skip
    return cmd + (["--clones", args.clones] if args.clones is not None else [])


def run_and_score(seq: Sequence, args: argparse.Namespace, out: Path) -> None:
    d = out / seq.name
    run(run_cmd(seq, args, d), log=out / f"{seq.name}.log")
    run([tool("msckf_eval"), "--gt", seq.gt, "--gt-format", "euroc", "--est", d / "est.txt", "--name", seq.name,
         "--out", d / "eval"], log=out / f"{seq.name}_eval.log")  # fmt: skip


MAX_ATE_FRACTION = 0.10  # of the distance travelled, else the run counts as diverged


def evaluate(report: GateReport, seq: Sequence, out: Path) -> bool:
    """Records the sequence's scores; returns whether the filter diverged."""
    d = out / seq.name
    status = read_csv(d / "run.csv")[0]
    m = read_csv(d / "eval" / "metrics.csv")[0]
    ms = float(np.mean([fnum(r["time_ms"]) for r in read_csv(d / "frames.csv")]))
    ate, rot, re = fnum(m["ate_rmse_m"]), fnum(m["ate_rot_rmse_deg"]), fnum(m["re_trans_mean_pct"])
    report.record(f"{seq.name}: ATE RMSE", ate, "lower", "m")
    report.record(f"{seq.name}: rotation ATE RMSE", rot, "lower", "deg")
    report.record(f"{seq.name}: relative translation error, mean", re, "lower", "%")
    report.record(f"{seq.name}: back end time per image, mean", ms, "lower", "ms")
    report.note(f"{seq.name}: ATE {ate:.3f} m over {fnum(m['length_m']):.0f} m, rotation {rot:.2f} deg, RE {re:.2f} %,"
                f" {ms:.1f} ms per image; init {status['init']}, {status['images']} images")  # fmt: skip
    return status["diverged"] == "1" or not ate <= MAX_ATE_FRACTION * fnum(m["length_m"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--lamaria", type=Path, default=large_data_dir() / "lamaria" / "training")
    ap.add_argument("--clones", type=int, help="overrides the filter's window size max_clones (diagnosis)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="stage3_real", help="gate_out subdirectory")
    args = ap.parse_args()

    out = gate_out(args.out)
    report = GateReport("Stage 3: MSCKF on real data")
    seqs = real_sequences(args, report)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in sequences.submit_tracking(ex, seqs):
            f.result()
        for f in [ex.submit(run_and_score, s, args, out) for s in seqs]:
            f.result()
    diverged = [s.name for s in seqs if evaluate(report, s, out)]
    report.check("sequences where the filter diverged (or ATE > 10 % of distance)", len(diverged), "<=", 0)
    if diverged:
        report.note(f"diverged: {', '.join(diverged)}")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
