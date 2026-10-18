"""Stage 3 gate, real data: the MSCKF on recorded sequences (plan, Stage 3
metrics: accuracy, robustness, runtime).

Runs tools/msckf_run on EuRoC MH_01-05, TUM-VI room1 and LaMAria R_01-03 (if
converted) with the Stage 1b front-end tracks (msckf_gates.sequences, cached
by the Stage 1b gate), the Stage 1b initialization and the same noise
inflation (IMU x4, 2 px), with FEJ (the filter's default) and without, then
scores each trajectory with tools/msckf_eval (ATE after a position + yaw
alignment, relative error).

Gated, with FEJ (plan, "Robustness: 0 divergences on all test sequences"):
every sequence initializes, runs to the end and never diverges: no
non-finite or runaway state (msckf_run), a trajectory msckf_eval can score,
and ATE RMSE at most 10 % of the distance travelled (a filter that has lost
track can stay finite: EuRoC MH_03 at 30 clones reached 586 m ATE over 131 m).
The same is reported without FEJ. Recorded for benchmark_compare, not gated
until the OpenVINS mono baseline exists (plan: "ATE within 20 % of OpenVINS
mono, run as a binary"): ATE RMSE, rotation ATE RMSE, mean relative
translation error, and the back end's time per image (measured with
--workers runs in parallel: compare across runs with the same setting only).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates import sequences
from msckf_gates.common import GateReport, data_dir, fnum, gate_out, large_data_dir, read_csv, run, tool
from msckf_gates.sequences import Sequence

MAX_ATE_FRACTION = 0.10  # of the distance travelled, else the run counts as diverged
Variant = tuple[Sequence, bool]  # sequence, FEJ


def real_sequences(args: argparse.Namespace, report: GateReport) -> list[Sequence]:
    lamaria = [args.lamaria / s for s in sequences.LAMARIA]
    missing = [d.name for d in lamaria if not (d / "gt_imu.csv").exists()]
    if missing:
        report.note(f"LaMAria not run, missing {', '.join(missing)}")
        lamaria = []
    return [*sequences.euroc(args.euroc), sequences.tumvi_room1(args.tumvi), *sequences.converted("lamaria", lamaria)]


def label(v: Variant) -> str:
    return f"{v[0].name}, {'FEJ' if v[1] else 'no FEJ'}"


def out_dir(out: Path, v: Variant) -> Path:
    return out / f"{v[0].name}_{'fej' if v[1] else 'nofej'}"


def run_cmd(v: Variant, args: argparse.Namespace, d: Path) -> list:
    seq, fej = v
    cmd = [tool("msckf_run"), *seq.cam, "--tracks", sequences.tracks_csv(seq), "--imu-yaml", seq.imu_yaml,
           "--gt", seq.gt, "--bias", seq.bias, *sequences.REAL_NOISE, "--out", d]  # fmt: skip
    cmd += [] if fej else ["--no-fej"]
    return cmd + (["--clones", args.clones] if args.clones is not None else [])


def run_and_score(v: Variant, args: argparse.Namespace, out: Path) -> str:
    """Runs the filter and msckf_eval; returns the failure, empty on success."""
    d = out_dir(out, v)
    try:
        run(run_cmd(v, args, d), log=d.parent / f"{d.name}.log")
        run([tool("msckf_eval"), "--gt", v[0].gt, "--gt-format", "euroc", "--est", d / "est.txt", "--name",
             v[0].name, "--out", d / "eval"], log=d.parent / f"{d.name}_eval.log")  # fmt: skip
    except RuntimeError as e:
        return str(e)
    return ""


def evaluate(report: GateReport, v: Variant, out: Path) -> bool:
    """Records the run's scores; returns whether it diverged."""
    d = out_dir(out, v)
    status = read_csv(d / "run.csv")[0]
    m = read_csv(d / "eval" / "metrics.csv")[0]
    ms = float(np.mean([fnum(r["time_ms"]) for r in read_csv(d / "frames.csv")]))
    ate, rot, re = fnum(m["ate_rmse_m"]), fnum(m["ate_rot_rmse_deg"]), fnum(m["re_trans_mean_pct"])
    report.record(f"{label(v)}: ATE RMSE", ate, "lower", "m")
    report.record(f"{label(v)}: rotation ATE RMSE", rot, "lower", "deg")
    report.record(f"{label(v)}: relative translation error, mean", re, "lower", "%")
    report.record(f"{label(v)}: back end time per image, mean", ms, "lower", "ms")
    report.note(f"{label(v)}: ATE {ate:.3f} m over {fnum(m['length_m']):.0f} m, rotation {rot:.2f} deg, "
                f"RE {re:.2f} %, {ms:.1f} ms per image; init {status['init']}, {status['images']} images")  # fmt: skip
    lost = not ate <= MAX_ATE_FRACTION * fnum(m["length_m"])
    return status["diverged"] == "1" or bool(m["error"]) or lost


def failed(report: GateReport, v: Variant, out: Path, error: str) -> bool:
    """Whether the run failed or diverged (noted)."""
    if error:
        report.note(f"{label(v)}: FAILED ({error})")
        return True
    return evaluate(report, v, out)


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
    variants: list[Variant] = [(s, fej) for fej in (True, False) for s in seqs]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in sequences.submit_tracking(ex, seqs):
            f.result()
        errors = list(ex.map(lambda v: run_and_score(v, args, out), variants))
    bad = [v for v, e in zip(variants, errors) if failed(report, v, out, e)]
    fej_bad = [v for v in bad if v[1]]
    report.check("sequences failed or diverged with FEJ (or ATE > 10 % of distance)", len(fej_bad), "<=", 0)
    if bad:
        report.note(f"failed or diverged: {', '.join(label(v) for v in bad)}")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
