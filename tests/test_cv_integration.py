"""Synthetic integration tests for fixed-manifest Constant Velocity evaluation."""

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

import numpy as np

from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.data.av2_dataset import write_cache_shards, write_json
from trajectory_prediction.evaluation.cv import (
    evaluate_cv_samples,
    load_evaluation_manifest,
    load_samples_from_manifest,
    run_cv_evaluation,
)


def make_global_cv_sample(scenario_id: str = "scenario-a") -> TrajectorySample:
    """Create a global val sample following exact one-metre/second motion."""

    return TrajectorySample(
        scenario_id=scenario_id,
        split="val",
        focal_track_id=f"focal-{scenario_id}",
        city="PIT",
        history_timesteps=np.asarray([0, 1, 2], dtype=np.int64),
        future_timesteps=np.asarray([3, 4], dtype=np.int64),
        timestamps_ns=np.asarray(
            [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
            dtype=np.int64,
        ),
        history_position=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
        ),
        history_velocity=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32
        ),
        history_heading=np.zeros(3, dtype=np.float32),
        history_observed=np.ones(3, dtype=bool),
        history_mask=np.ones(3, dtype=bool),
        future_position=np.asarray([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        future_velocity=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        future_heading=np.zeros(2, dtype=np.float32),
        future_observed=np.zeros(2, dtype=bool),
        future_mask=np.ones(2, dtype=bool),
        origin_xy=np.asarray([2.0, 0.0], dtype=np.float32),
        reference_heading_rad=0.0,
    )


def write_synthetic_manifest(
    root: Path,
    samples: list[TrajectorySample],
) -> Path:
    """Write cache shards and the same format_version=1 manifest as preprocessing."""

    shards = write_cache_shards(samples, root / "cache", shard_size=2)
    manifest = {
        "format_version": 1,
        "dataset": "argoverse2_motion_forecasting",
        "split": "val",
        "sampling": {
            "method": "seeded_random_without_replacement",
            "seed": 2026,
            "requested_scene_count": len(samples),
            "available_scene_count": len(samples),
            "ordering": "scenario_id_ascending_after_sampling",
        },
        "window": {
            "history_steps": 3,
            "future_steps": 2,
            "timestamp_source": "synthetic integer nanoseconds",
        },
        "scenario_ids": [sample.scenario_id for sample in samples],
        "source_files": [
            f"{sample.scenario_id}/scenario_{sample.scenario_id}.parquet"
            for sample in samples
        ],
        "shards": shards,
    }
    return write_json(root / "manifest.json", manifest)


class CVIntegrationTest(unittest.TestCase):
    def test_manifest_to_metrics_and_saved_predictions_closes_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_synthetic_manifest(
                root / "prepared",
                [make_global_cv_sample("scenario-a"), make_global_cv_sample("scenario-b")],
            )
            output = root / "evaluation"

            summary, paths = run_cv_evaluation(manifest, output)

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["failed_scene_count"], 0)
            self.assertEqual(summary["ADE"], 0.0)
            self.assertEqual(summary["FDE"], 0.0)
            self.assertEqual(summary["MissRate"], 0.0)
            self.assertEqual(len(str(summary["manifest_sha256"])), 64)
            self.assertEqual(
                set(paths),
                {
                    "summary",
                    "metrics",
                    "prediction_index",
                    "predictions",
                    "failures",
                    "run_config",
                    "environment",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))

            with paths["metrics"].open(encoding="utf-8", newline="") as stream:
                metric_rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["scenario_id"] for row in metric_rows],
                ["scenario-a", "scenario-b"],
            )
            self.assertEqual([float(row["ADE"]) for row in metric_rows], [0.0, 0.0])

            with np.load(paths["predictions"], allow_pickle=False) as payload:
                self.assertEqual(payload["local_prediction_m"].shape, (2, 2, 2))
                self.assertEqual(payload["global_prediction_m"].shape, (2, 2, 2))
                np.testing.assert_allclose(
                    payload["local_prediction_m"][0], [[1.0, 0.0], [2.0, 0.0]]
                )
                np.testing.assert_allclose(
                    payload["global_prediction_m"][0], [[3.0, 0.0], [4.0, 0.0]]
                )
                np.testing.assert_array_equal(
                    payload["future_timestamps_ns"][0],
                    [3_000_000_000, 4_000_000_000],
                )

            saved_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(saved_summary, summary)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_synthetic_manifest(
                root / "prepared", [make_global_cv_sample()]
            )
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                run_cv_evaluation(manifest, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_manifest_order_must_match_loaded_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_synthetic_manifest(
                root,
                [make_global_cv_sample("scenario-a"), make_global_cv_sample("scenario-b")],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scenario_ids"] = list(reversed(manifest["scenario_ids"]))
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "scenario order"):
                load_samples_from_manifest(manifest_path)

    def test_manifest_rejects_non_validation_or_duplicate_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_synthetic_manifest(
                root, [make_global_cv_sample("scenario-a")]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["split"] = "train"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "official val split"):
                load_evaluation_manifest(manifest_path)

            manifest["split"] = "val"
            manifest["scenario_ids"] = ["scenario-a", "scenario-a"]
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_evaluation_manifest(manifest_path)

    def test_failed_scene_is_counted_without_weighting_successful_metrics(self) -> None:
        good = make_global_cv_sample("good")
        bad = replace(
            make_global_cv_sample("bad"),
            history_mask=np.zeros(3, dtype=bool),
        )

        result = evaluate_cv_samples([good, bad])

        self.assertEqual(result.summary["requested_scene_count"], 2)
        self.assertEqual(result.summary["sample_count"], 1)
        self.assertEqual(result.summary["failed_scene_count"], 1)
        self.assertEqual(result.summary["ADE"], 0.0)
        self.assertEqual(result.failures[0]["scenario_id"], "bad")

    def test_all_failed_scenes_are_rejected(self) -> None:
        bad = replace(
            make_global_cv_sample("bad"),
            history_mask=np.zeros(3, dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "All 1"):
            evaluate_cv_samples([bad])

    def test_future_label_changes_do_not_change_cv_predictions(self) -> None:
        original = make_global_cv_sample("original")
        changed = replace(
            make_global_cv_sample("changed"),
            future_position=np.asarray([[100.0, -50.0], [-20.0, 80.0]]),
            future_velocity=np.full((2, 2), 999.0),
            future_heading=np.asarray([1.0, -2.0]),
            future_mask=np.asarray([True, False]),
        )

        original_result = evaluate_cv_samples([original])
        changed_result = evaluate_cv_samples([changed])

        np.testing.assert_array_equal(
            original_result.local_predictions_m[0],
            changed_result.local_predictions_m[0],
        )
        np.testing.assert_array_equal(
            original_result.global_predictions_m[0],
            changed_result.global_predictions_m[0],
        )

    def test_missing_cache_requires_explicit_raw_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_synthetic_manifest(
                root, [make_global_cv_sample("scenario-a")]
            )
            for shard in (root / "cache").iterdir():
                shard.rename(root / f"hidden-{shard.name}")

            with self.assertRaisesRegex(FileNotFoundError, "--data-root"):
                load_samples_from_manifest(manifest_path)

    def test_evaluate_cli_runs_on_synthetic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_synthetic_manifest(
                root / "prepared", [make_global_cv_sample()]
            )
            output = root / "cli-output"
            project_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root / "src")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "evaluate.py"),
                    "--model",
                    "cv",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Evaluated 1 AV2 val scenes", completed.stdout)
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
