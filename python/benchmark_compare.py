#!/usr/bin/env python3
"""Compares a benchmark run with a baseline run and flags degradations.

A run is a directory written by scripts/run_benchmarks.sh:
<root>/<date-time>/<benchmark>/metrics.json, one per benchmark (unit tests,
each gate, tracker_compare). Each metrics.json holds {"passed", "metrics":
[{name, value, better}]}, with better "lower", "higher" or null (recorded,
not judged).

Flags, per benchmark:
  REGRESSION  the benchmark (a gate) passed in the baseline and fails now, or a
              gated check went from pass to fail
  DEGRADED    a metric moved in its worse direction by more than --rel-tol of
              the baseline value (and more than --abs-tol)
  IMPROVED    the same, in the better direction
  NEW / GONE  metrics present in only one run

    python python/benchmark_compare.py RUN [--baseline RUN0] [--rel-tol 0.02]

The baseline defaults to the latest earlier run in the same root that has
the benchmark. Writes RUN/compare.md; exits 1 if anything regressed or
degraded.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    d: dict[str, Any] = json.loads(path.read_text())
    return d


def previous_run(run: Path, benchmark: str) -> Path | None:
    """Latest run before `run` (run directories sort by their date-time name) with this benchmark."""
    earlier = sorted(p for p in run.parent.iterdir() if p.is_dir() and p.name < run.name)
    for p in reversed(earlier):
        if (p / benchmark / "metrics.json").exists():
            return p
    return None


def compare(new: dict[str, Any], old: dict[str, Any], rel_tol: float, abs_tol: float) -> list[tuple[str, str, str]]:
    """[(flag, metric, detail)] for one benchmark."""
    out = []
    if old.get("passed") is True and new.get("passed") is False:
        out.append(("REGRESSION", "(benchmark)", "passed in the baseline, fails now"))
    olds = {m["name"]: m for m in old["metrics"]}
    news = {m["name"]: m for m in new["metrics"]}
    for name, m in news.items():
        if name not in olds:
            out.append(("NEW", name, f"{m['value']}"))
            continue
        o = olds[name]
        if o.get("passed") is True and m.get("passed") is False:
            out.append(
                (
                    "REGRESSION",
                    name,
                    f"{o['value']:.4g} -> {m['value']:.4g} (threshold {m.get('op', '')} " f"{m.get('threshold')})",
                )
            )
            continue
        a, b = o["value"], m["value"]
        if a is None or b is None or m.get("better") is None:
            continue
        delta = b - a
        if abs(delta) <= max(abs_tol, rel_tol * abs(a)):
            continue
        worse = delta > 0 if m["better"] == "lower" else delta < 0
        rel = f" ({100 * delta / abs(a):+.1f} %)" if a != 0 and math.isfinite(a) else ""
        out.append(("DEGRADED" if worse else "IMPROVED", name, f"{a:.4g} -> {b:.4g}{rel}"))
    for name in olds.keys() - news.keys():
        out.append(("GONE", name, ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--baseline", type=Path, help="baseline run (default: latest earlier run per benchmark)")
    ap.add_argument("--rel-tol", type=float, default=0.02, help="relative change ignored (default 2 %%)")
    ap.add_argument("--abs-tol", type=float, default=1e-9)
    args = ap.parse_args()

    run = args.run.resolve()
    order = {"REGRESSION": 0, "DEGRADED": 1, "IMPROVED": 2, "NEW": 3, "GONE": 4}
    md = [f"# Benchmark comparison: {run.name}", ""]
    bad = False
    for bench in sorted(p.parent.name for p in run.glob("*/metrics.json")):
        new = load(run / bench / "metrics.json")
        base = args.baseline.resolve() if args.baseline else previous_run(run, bench)
        status = {True: "PASS", False: "FAIL", None: "-"}[new.get("passed")]
        md += [f"## {bench}: {status}", ""]
        if base is None or not (base / bench / "metrics.json").exists():
            md += ["No baseline run; recorded only.", ""]
            continue
        flags = sorted(
            compare(new, load(base / bench / "metrics.json"), args.rel_tol, args.abs_tol), key=lambda f: order[f[0]]
        )
        counts = {k: sum(f[0] == k for f in flags) for k in order}
        md += [f"Baseline `{base.name}`: " + ", ".join(f"{n} {k.lower()}" for k, n in counts.items()), ""]
        shown = [f for f in flags if f[0] in ("REGRESSION", "DEGRADED", "IMPROVED")]
        if shown:
            md += ["| Flag | Metric | Change |", "| --- | --- | --- |"]
            md += [f"| {f} | {name} | {detail} |" for f, name, detail in shown]
            md += [""]
        bad |= counts["REGRESSION"] + counts["DEGRADED"] > 0
    md += [
        f"**{'DEGRADATIONS FOUND' if bad else 'No regressions or degradations'}** "
        f"(relative tolerance {100 * args.rel_tol:g} %)."
    ]
    (run / "compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\ncompare: {run / 'compare.md'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
