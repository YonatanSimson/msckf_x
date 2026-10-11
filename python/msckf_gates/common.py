"""Shared helpers for the gate checkers: paths, running the C++ tools, and a
pass/fail table that is printed and written to <out>/report.md."""
import csv
import json
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


# The adopted front end (plan, Stage 1, "Head-to-head against Basalt"): our
# vendored Basalt flow (tracker_eval --tracker basalt) with Basalt's shipped
# optical-flow settings (data/tumvi_512_config.json and data/euroc_config.json
# at Basalt 914808d: frame_to_frame, pattern 51, 5 iterations, 3 levels, 0.2 px
# forward-backward; detection grid 40 px for TUM-VI 512, 50 px otherwise),
# plus the camera model's valid-region mask so every feature has a bearing.
def basalt_flow_args(grid_px):
    return ["--tracker", "basalt", "--basalt-type", "frame_to_frame", "--grid", str(grid_px), "--fb-dist2", "0.04",
            "--pattern", "51", "--iterations", "5", "--levels", "3", "--mask-valid"]


FRONT_END_TUMVI = basalt_flow_args(40)
FRONT_END_EUROC = basalt_flow_args(50)  # also used for TartanAir and KITTI


def euroc_images(cam_dir):
    """[(t_ns, path)] of a EuRoC-format camera directory (mav0/camN); empty if absent."""
    cam_dir = Path(cam_dir)
    out = []
    if not (cam_dir / "data.csv").exists():
        return out
    with open(cam_dir / "data.csv") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t, name = line.strip().split(",")
            out.append((int(t), cam_dir / "data" / name.strip()))
    return out


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

    def metrics(self):
        """Machine-readable checks for python/benchmark_compare.py: each value,
        its threshold, and which direction is better (from the check's op)."""
        return [{"name": c.name, "value": c.value, "unit": c.unit, "op": c.op, "threshold": c.threshold,
                 "better": "lower" if c.op in ("<=", "<") else "higher", "passed": c.passed}
                for c in self.checks]

    def finish(self, out_dir):
        md = self.markdown()
        (Path(out_dir) / "report.md").write_text(md)
        write_metrics(out_dir, self.metrics(), passed=self.passed)
        print()
        print(md)
        print(f"report: {Path(out_dir) / 'report.md'}")
        return 0 if self.passed else 1


def write_metrics(out_dir, metrics, passed=None):
    """<out_dir>/metrics.json: {"passed": bool|null, "metrics": [{name, value,
    better: "lower"|"higher"|null, ...}]}, read by python/benchmark_compare.py."""

    def plain(v):  # numpy scalars -> Python; NaN/inf -> null
        if hasattr(v, "item"):
            v = v.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v

    clean = [{k: plain(v) for k, v in m.items()} for m in metrics]
    doc = {"passed": plain(passed), "metrics": clean}
    (Path(out_dir) / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")


def nees_band(n, dof):
    """Two-sided 95 % band of the mean NEES of n independent chi2(dof) samples
    (Wilson-Hilferty approximation of the chi2(n dof) quantiles, / n)."""
    k = n * dof
    c = 2.0 / (9.0 * k)
    return k * (1 - c - 1.96 * math.sqrt(c)) ** 3 / n, k * (1 - c + 1.96 * math.sqrt(c)) ** 3 / n
