"""Tests for python/benchmark_compare.py's baseline lookup.

cd python && python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

from benchmark_compare import compare, resolve_baseline


class ResolveBaselineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "2026-10-17_145125"
        self.run.mkdir()
        (self.root / "2026-10-17_124932").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_bare_name_resolves_next_to_the_run(self):
        found = resolve_baseline(Path("2026-10-17_124932"), self.run)

        self.assertEqual(found, (self.root / "2026-10-17_124932").resolve())

    def test_full_path_is_used_as_given(self):
        found = resolve_baseline(self.root / "2026-10-17_124932", self.run)

        self.assertEqual(found, (self.root / "2026-10-17_124932").resolve())

    def test_missing_baseline_is_an_error(self):
        with self.assertRaises(SystemExit):
            resolve_baseline(Path("2026-01-01_000000"), self.run)


def run_of(value):
    return {"metrics": [{"name": "gravity RMSE", "value": value, "better": "lower"}]}


class CompareNanTest(unittest.TestCase):
    def test_metric_becoming_nan_is_degraded(self):
        flags = compare(run_of(float("nan")), run_of(3.4), 0.02, 1e-9)

        self.assertEqual([f[0] for f in flags], ["DEGRADED"])

    def test_metric_recovering_from_nan_is_improved(self):
        flags = compare(run_of(3.4), run_of(float("nan")), 0.02, 1e-9)

        self.assertEqual([f[0] for f in flags], ["IMPROVED"])

    def test_nan_in_both_runs_is_not_flagged(self):
        flags = compare(run_of(float("nan")), run_of(float("nan")), 0.02, 1e-9)

        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
