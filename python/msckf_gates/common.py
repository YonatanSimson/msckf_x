"""Shared helpers for the gate checkers: paths, running the C++ tools, and a
pass/fail table that is printed and written to <out>/report.md."""
import csv
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build_dir():
    return Path(os.environ.get("MSCKF_BUILD", REPO / "build"))


def data_dir():
    """Small datasets (WSL disk): $MSCKF_DATA, default ~/data."""
    return Path(os.environ.get("MSCKF_DATA", Path.home() / "data"))


def large_data_dir():
    """Large datasets: $MSCKF_DATA_LARGE, default /mnt/c/Users/user/data."""
    return Path(os.environ.get("MSCKF_DATA_LARGE", "/mnt/c/Users/user/data"))


def gate_out(name):
    out = Path(os.environ.get("MSCKF_GATE_OUT", REPO / "gate_out")) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def tool(name):
    path = build_dir() / name
    if not path.exists():
        sys.exit(f"{path} not found; build first (cmake --build {build_dir()})")
    return str(path)


def run(cmd, log=None):
    """Runs a command, echoing it; returns stdout. Raises on failure."""
    cmd = [str(c) for c in cmd]
    print("$", shlex.join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(f"$ {shlex.join(cmd)}\n{res.stdout}{res.stderr}")
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise RuntimeError(f"command failed ({res.returncode}): {shlex.join(cmd)}")
    return res.stdout


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else math.nan


@dataclass
class Check:
    name: str
    value: float
    op: str  # "<=", ">=", "<", ">"
    threshold: float
    unit: str = ""

    @property
    def passed(self):
        v, t = self.value, self.threshold
        if math.isnan(v):
            return False
        return {"<=": v <= t, ">=": v >= t, "<": v < t, ">": v > t}[self.op]


@dataclass
class GateReport:
    title: str
    checks: list = field(default_factory=list)
    notes: list = field(default_factory=list)  # informational lines (not gated)

    def check(self, name, value, op, threshold, unit=""):
        self.checks.append(Check(name, value, op, threshold, unit))

    def note(self, line):
        self.notes.append(line)

    @property
    def passed(self):
        return bool(self.checks) and all(c.passed for c in self.checks)

    def markdown(self):
        lines = [f"# {self.title}", "", f"**{'PASS' if self.passed else 'FAIL'}**", "",
                 "| Check | Value | Threshold | Result |", "| --- | --- | --- | --- |"]
        for c in self.checks:
            lines.append(f"| {c.name} | {c.value:.4g} {c.unit} | {c.op} {c.threshold:.4g} {c.unit} "
                         f"| {'pass' if c.passed else 'FAIL'} |")
        if self.notes:
            lines += ["", "Not gated:", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"

    def finish(self, out_dir):
        md = self.markdown()
        (Path(out_dir) / "report.md").write_text(md)
        print()
        print(md)
        print(f"report: {Path(out_dir) / 'report.md'}")
        return 0 if self.passed else 1
