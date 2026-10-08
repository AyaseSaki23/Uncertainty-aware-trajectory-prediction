"""Tests for logical reproducibility checks across saved CV runs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_cv_integration import make_global_cv_sample, write_synthetic_manifest
from trajectory_prediction.evaluation.cv import run_cv_evaluation
from trajectory_prediction.evaluation.reproducibility import (
    ReproducibilityError,
    compare_cv_runs,
)


def _make_two_runs(root: Path) -> tuple[Path, Path, Path]:
    manifest = write_synthetic_manifest(
        root / "prepared",
        [make_global_cv_sample("scenario-a"), make_global_cv_sample("scenario-b")],
    )
    run_a = root / "run-a"
    run_b = root / "run-b"
    run_cv_evaluation(manifest, run_a)
    run_cv_evaluation(manifest, run_b)
    return manifest, run_a, run_b


class ReproducibilityTest(unittest.TestCase):
    def test_independent_runs_match_logically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, run_a, run_b = _make_two_runs(Path(directory))

            report = compare_cv_runs(run_a, run_b)

            self.assertTrue(report["matched"])
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["absolute_tolerance"], 1e-12)
            self.assertTrue(report["ignored_container_bytes"])
            for detail in report["prediction_arrays"].values():
                if "max_abs_difference" in detail:
                    self.assertEqual(detail["max_abs_difference"], 0.0)

    def test_npz_container_bytes_are_not_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, run_a, run_b = _make_two_runs(Path(directory))
            predictions = run_b / "predictions.npz"
            original_bytes = predictions.read_bytes()
            with np.load(predictions, allow_pickle=False) as payload:
                arrays = {key: payload[key].copy() for key in reversed(payload.files)}
            np.savez(predictions, **arrays)

            self.assertNotEqual(predictions.read_bytes(), original_bytes)
            report = compare_cv_runs(run_a, run_b)
            self.assertTrue(report["matched"])

    def test_changed_prediction_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, run_a, run_b = _make_two_runs(Path(directory))
            predictions = run_b / "predictions.npz"
            with np.load(predictions, allow_pickle=False) as payload:
                arrays = {key: payload[key].copy() for key in payload.files}
            arrays["local_prediction_m"][0, 0, 0] += 0.01
            np.savez_compressed(predictions, **arrays)

            with self.assertRaisesRegex(
                ReproducibilityError, "max absolute difference"
            ):
                compare_cv_runs(run_a, run_b)

    def test_changed_metric_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, run_a, run_b = _make_two_runs(Path(directory))
            metrics = run_b / "per_scene_metrics.csv"
            with metrics.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                fieldnames = list(rows[0])
            rows[0]["ADE"] = "0.5"
            with metrics.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ReproducibilityError, "ADE differs"):
                compare_cv_runs(run_a, run_b)

    def test_changed_manifest_fails_recorded_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, run_a, run_b = _make_two_runs(Path(directory))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["sampling"]["seed"] = 999
            manifest.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(
                ReproducibilityError, "manifest_sha256 does not match"
            ):
                compare_cv_runs(run_a, run_b)

    def test_cli_writes_comparison_report_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, run_a, run_b = _make_two_runs(root)
            report_path = root / "comparison.json"
            project_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root / "src")
            command = [
                sys.executable,
                str(project_root / "scripts" / "compare_cv_runs.py"),
                "--run-a",
                str(run_a),
                "--run-b",
                str(run_b),
                "--output",
                str(report_path),
            ]

            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Logical reproducibility check passed", completed.stdout)
            self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["matched"])

            repeated = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("will not be overwritten", repeated.stderr)

    def test_absolute_tolerance_must_be_finite_and_non_negative(self) -> None:
        for value in (-1.0, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    compare_cv_runs("unused-a", "unused-b", atol=value)


if __name__ == "__main__":
    unittest.main()
