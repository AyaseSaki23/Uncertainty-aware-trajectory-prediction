"""Tests for CV review figures and structured baseline result tables."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tests.test_cv_integration import make_global_cv_sample, write_synthetic_manifest
from trajectory_prediction.evaluation.cv import evaluate_cv_samples, run_cv_evaluation
from trajectory_prediction.reporting.tables import generate_cv_results_tables
from trajectory_prediction.visualization.bev import plot_prediction_comparison
from trajectory_prediction.visualization.results import generate_review_visualizations


def make_review_samples() -> list[object]:
    """Create distinct straight, turning, and high-error CV cases."""

    straight = make_global_cv_sample("straight")
    turn = replace(
        make_global_cv_sample("turn"),
        future_position=np.asarray([[2.7, 0.6], [2.7, 1.7]], dtype=np.float32),
        future_heading=np.asarray([np.pi / 4, np.pi / 2], dtype=np.float32),
    )
    failure = replace(
        make_global_cv_sample("failure"),
        future_position=np.asarray([[2.0, -8.0], [2.0, -18.0]], dtype=np.float32),
        future_heading=np.zeros(2, dtype=np.float32),
    )
    return [straight, turn, failure]


def create_review_run(root: Path) -> Path:
    """Create one synthetic three-scene evaluation run."""

    manifest = write_synthetic_manifest(root / "prepared", make_review_samples())
    run_dir = root / "run"
    run_cv_evaluation(manifest, run_dir)
    return run_dir


class CVVisualizationAndTableTest(unittest.TestCase):
    def test_review_selection_generates_three_distinct_dual_view_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_review_run(root)
            output_dir = root / "review"

            result = generate_review_visualizations(run_dir, output_dir)

            self.assertEqual(result["generated_scene_count"], 3)
            categories = [case["review_category"] for case in result["cases"]]
            scenario_ids = [case["scenario_id"] for case in result["cases"]]
            self.assertEqual(categories, ["straight", "turn", "cv_failure"])
            self.assertEqual(scenario_ids, ["straight", "turn", "failure"])
            self.assertEqual(len(set(scenario_ids)), 3)
            for filename in result["generated_files"]:
                path = output_dir / filename
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 5_000)
                image = plt.imread(path)
                self.assertGreater(image.shape[1], image.shape[0])
            self.assertTrue((output_dir / "review_manifest.json").is_file())
            checklist = (output_dir / "review_checklist.md").read_text(encoding="utf-8")
            self.assertIn("straight", checklist)
            self.assertIn("cv_failure", checklist)

    def test_masked_paths_are_drawn_as_separate_contiguous_segments(self) -> None:
        sample = replace(
            make_global_cv_sample("missing"),
            history_position=np.asarray(
                [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32
            ),
            history_velocity=np.asarray(
                [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32
            ),
            history_observed=np.asarray([True, False, True]),
            history_mask=np.asarray([True, False, True]),
            future_position=np.asarray([[3.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            future_velocity=np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            future_mask=np.asarray([True, False]),
        )
        evaluation = evaluate_cv_samples([sample])
        figure = plot_prediction_comparison(
            sample,
            evaluation.local_predictions_m[0],
            evaluation.global_predictions_m[0],
            evaluation.rows[0],
        )
        try:
            self.assertEqual(len(figure.axes), 2)
            for axis in figure.axes:
                history_segments = [line for line in axis.lines if line.get_gid() == "history"]
                truth_segments = [
                    line for line in axis.lines if line.get_gid() == "ground_truth"
                ]
                predictions = [
                    line for line in axis.lines if line.get_gid() == "cv_prediction"
                ]
                self.assertEqual(len(history_segments), 2)
                self.assertEqual(len(truth_segments), 1)
                self.assertEqual(len(predictions), 1)
                self.assertIn("(m)", axis.get_xlabel())
                self.assertEqual(float(axis.get_aspect()), 1.0)
            self.assertIn("ADE=", figure._suptitle.get_text())
            self.assertIn("FDE=", figure._suptitle.get_text())
        finally:
            plt.close(figure)

    def test_manual_selection_and_existing_output_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_review_run(root)
            output_dir = root / "manual"

            result = generate_review_visualizations(
                run_dir, output_dir, scenario_ids=["turn"]
            )
            self.assertEqual(result["generated_scene_count"], 1)
            self.assertEqual(result["cases"][0]["review_category"], "manual")
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                generate_review_visualizations(run_dir, output_dir)
            with self.assertRaisesRegex(ValueError, "not in the run"):
                generate_review_visualizations(
                    run_dir, root / "unknown", scenario_ids=["does-not-exist"]
                )

    def test_result_table_recomputes_and_labels_fixed_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_review_run(root)
            output_dir = root / "tables"

            paths = generate_cv_results_tables([run_dir], output_dir)

            with paths["csv"].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["evaluation_scope"], "fixed_validation_subset")
            self.assertEqual(rows[0]["scene_count"], "3")
            self.assertEqual(rows[0]["failed_scene_count"], "0")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(rows[0]["ADE_m"]), summary["ADE"])
            self.assertAlmostEqual(float(rows[0]["FDE_m"]), summary["FDE"])
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Fixed AV2 Validation Subset", markdown)
            self.assertIn("不是 AV2 官方完整测试集结果", markdown)
            self.assertIn("ADE (m)", markdown)
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                generate_cv_results_tables([run_dir], output_dir)

    def test_result_table_rejects_summary_per_scene_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_review_run(root)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["ADE"] = float(summary["ADE"]) + 1.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ADE mismatch"):
                generate_cv_results_tables([run_dir], root / "tables")

    def test_visualization_and_table_clis_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_review_run(root)
            project_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root / "src")

            visualize = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "visualize_predictions.py"),
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(root / "cli-review"),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(visualize.returncode, 0, visualize.stderr)
            self.assertIn("Generated 3 review figures", visualize.stdout)

            tables = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "make_paper_tables.py"),
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(root / "cli-tables"),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tables.returncode, 0, tables.stderr)
            self.assertIn("markdown:", tables.stdout)
            self.assertTrue((root / "cli-tables" / "cv_baseline_results.md").is_file())


if __name__ == "__main__":
    unittest.main()
