"""Stage 3 gate, real data: the MSCKF on recorded sequences (plan, Stage 3
metrics: accuracy, robustness, runtime).

Runs tools/msckf_run on every sequence on disk (msckf_gates.sequences: EuRoC
MH_01-05 and V1/V2_01-03, TUM-VI rooms, corridors and slides, the UZH-FPV Snapdragon sequences
with ground truth, LaMAria R_01-03 on the undistorted pinhole export and on
the raw fisheye) with the Stage 1b front-end tracks (cached by the Stage 1b
gate), the Stage 1b initialization and the same noise inflation (IMU x4,
2 px), without FEJ (the user's setting for the first test sweep, plan
Handover) and with it, then scores each trajectory with tools/msckf_eval
(ATE after a position + yaw alignment, relative error).

Gated, without FEJ (plan, "Robustness: 0 divergences on all test sequences"):
every sequence initializes, runs to the end and never diverges: no
non-finite or runaway state (msckf_run), a trajectory msckf_eval can score,
and ATE RMSE at most 10 % of the distance travelled (a filter that has lost
track can stay finite: EuRoC MH_03 at 30 clones reached 586 m ATE over 131 m).
The same is reported with FEJ. Recorded for benchmark_compare, not gated
until the OpenVINS mono baseline runs here (plan: "ATE within 20 % of
OpenVINS mono, run as a binary"): ATE RMSE, rotation ATE RMSE, mean relative
translation error, the back end's time per image (measured with --workers
runs in parallel: compare across runs with the same setting only), and on
EuRoC the ratio of our ATE to the published OpenVINS and sqrtVINS mono ATE
(msckf_gates.baselines), per sequence and as a geometric mean.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates import baselines, sequences
from msckf_gates.common import GateReport, data_dir, fnum, gate_out, large_data_dir, read_csv, run, tool
from msckf_gates.sequences import Sequence

MAX_ATE_FRACTION = 0.10  # of the distance travelled, else the run counts as diverged
Variant = tuple[Sequence, bool]  # sequence, FEJ


def present(dirs: list[Path], report: GateReport, what: str) -> list[Path]:
    """The converted sequence directories (gt_imu.csv exists); the missing ones are noted."""
    missing = [d.parent.name if d.name == "fisheye" else d.name for d in dirs if not (d / "gt_imu.csv").exists()]
    if missing:
        report.note(f"{what} not run, missing {', '.join(missing)}")
    return [d for d in dirs if (d / "gt_imu.csv").exists()]


def on_disk(seqs: list[Sequence], report: GateReport, what: str) -> list[Sequence]:
    """The sequences whose ground truth exists; the missing ones are noted."""
    missing = [s.name for s in seqs if not s.gt.exists()]
    if missing:
        report.note(f"{what} not run, missing {', '.join(missing)}")
    return [s for s in seqs if s.gt.exists()]


def real_sequences(args: argparse.Namespace, report: GateReport) -> list[Sequence]:
    lamaria = [args.lamaria / s for s in sequences.LAMARIA]
    uzh = present([args.uzh_fpv / f"{n}_snapdragon_with_gt" for n in sequences.UZH_FPV], report, "UZH-FPV")
    fisheye = present([d / "fisheye" for d in lamaria], report, "LaMAria fisheye")
    return [
        *sequences.euroc(args.euroc),
        *on_disk(sequences.tumvi(args.tumvi), report, "TUM-VI"),
        *sequences.converted("uzh", uzh),
        *sequences.converted("lamaria", present(lamaria, report, "LaMAria")),
        *sequences.lamaria_fisheye([d.parent for d in fisheye]),
    ]


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


def record_baseline_ratios(report: GateReport, v: Variant, ate: float) -> dict[str, float]:
    """Our ATE over each baseline's: the published mono ATE of the sequence
    (msckf_gates.baselines) and the ATE measured by the stage3_baselines
    benchmark, when present. {} when neither is known."""
    ratios: dict[str, float] = {}
    quoted = []
    for system, p in baselines.published(v[0].name).items():
        ratios[system] = ate / p.ate_m
        report.record(f"{label(v)}: ATE RMSE / {system} mono ATE (published)", ratios[system], "lower")
        quoted.append(f"{system} {p.ate_m:.2f} m published ({ratios[system]:.1f}x)")
    for system in baselines.SYSTEMS:
        measured = baselines.measured(system, v[0].name)
        if measured is not None:
            ratios[f"{system} measured"] = ate / measured
            report.record(f"{label(v)}: ATE RMSE / {system} mono ATE (measured)", ate / measured, "lower")
            quoted.append(f"{system} {measured:.3f} m measured ({ate / measured:.1f}x)")
    if quoted:
        report.note(f"{label(v)}: ATE vs mono baselines " + ", ".join(quoted))
    return ratios


def evaluate(report: GateReport, v: Variant, out: Path) -> tuple[bool, dict[str, float]]:
    """Records the run's scores; returns whether it diverged and its published-baseline ratios."""
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
    diverged = status["diverged"] == "1" or bool(m["error"]) or lost
    return diverged, record_baseline_ratios(report, v, ate)


def score(report: GateReport, v: Variant, out: Path, error: str) -> tuple[bool, dict[str, float]]:
    """Whether the run failed or diverged (noted), and its published-baseline ratios."""
    if error:
        report.note(f"{label(v)}: FAILED ({error})")
        return True, {}
    return evaluate(report, v, out)


def record_geometric_means(report: GateReport, ratios: list[dict[str, float]], fej: bool) -> None:
    """Geometric mean of our ATE / baseline ATE per system: over the EuRoC
    sequences for the published ATEs, over every sequence both ran for the
    measured ones."""
    mode = "FEJ" if fej else "no FEJ"
    for system in baselines.SYSTEMS:
        for kind, key, scope in (("published", system, "EuRoC"), ("measured", f"{system} measured", "all sequences")):
            values = [r[key] for r in ratios if key in r]
            if values:
                gm = float(np.exp(np.mean(np.log(values))))
                report.record(f"{scope}, {mode}: ATE RMSE / {system} mono ATE ({kind}), geometric mean", gm, "lower")
                report.note(f"{scope}, {mode}: {len(values)} sequences, ATE {gm:.2f}x {system} mono ({kind})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--uzh-fpv", type=Path, default=large_data_dir() / "uzh_fpv")
    ap.add_argument("--lamaria", type=Path, default=large_data_dir() / "lamaria" / "training")
    ap.add_argument("--clones", type=int, help="overrides the filter's window size max_clones (diagnosis)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="stage3_real", help="gate_out subdirectory")
    args = ap.parse_args()

    out = gate_out(args.out)
    report = GateReport("Stage 3: MSCKF on real data")
    seqs = real_sequences(args, report)
    variants: list[Variant] = [(s, fej) for fej in (False, True) for s in seqs]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in sequences.submit_tracking(ex, seqs):
            f.result()
        errors = list(ex.map(lambda v: run_and_score(v, args, out), variants))
    results = [(v, *score(report, v, out, e)) for v, e in zip(variants, errors)]
    bad = [v for v, diverged, _ in results if diverged]
    for fej in (False, True):
        record_geometric_means(report, [r for v, _, r in results if v[1] == fej], fej)
    nofej_bad = [v for v in bad if not v[1]]
    report.check("sequences failed or diverged without FEJ (or ATE > 10 % of distance)", len(nofej_bad), "<=", 0)
    if bad:
        report.note(f"failed or diverged: {', '.join(label(v) for v in bad)}")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
