"""Unit tests for AV2 parsing, causal slicing, masks, and cache reproducibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trajectory_prediction.data.av2_dataset import (
    ScenarioValidationError,
    build_focal_sample,
    load_cache_shard,
    read_av2_scenario,
    select_scenario_paths,
    validate_scenario_dataframe,
    write_cache_shards,
)
from trajectory_prediction.visualization.bev import plot_focal_sample_bev


SCENARIO_ID = "0000b0f9-99f9-4a1f-a231-5be9e4c523f7"


def make_frame(*, missing_focal_timestep: int | None = None) -> pd.DataFrame:
    """Create an official-schema synthetic scenario with a non-nominal real dt."""

    rows: list[dict[str, object]] = []
    num_timestamps = 110
    start_timestamp = 1_000_000_000
    dt_ns = 120_000_000
    end_timestamp = start_timestamp + (num_timestamps - 1) * dt_ns
    for track_id, limit in (("focal-1", num_timestamps), ("context-1", 50)):
        for timestep in range(limit):
            if track_id == "focal-1" and timestep == missing_focal_timestep:
                continue
            rows.append(
                {
                    "observed": timestep < 50,
                    "track_id": track_id,
                    "object_type": "vehicle",
                    "object_category": 3 if track_id == "focal-1" else 2,
                    "timestep": timestep,
                    "position_x": float(timestep),
                    "position_y": float(timestep * 2),
                    "heading": 0.25,
                    "velocity_x": 1.0,
                    "velocity_y": 2.0,
                    "scenario_id": SCENARIO_ID,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "num_timestamps": num_timestamps,
                    "focal_track_id": "focal-1",
                    "city": "PIT",
                }
            )
    return pd.DataFrame(rows)


class AV2DataPipelineTest(unittest.TestCase):
    def test_extracts_fields_actual_dt_and_causal_windows(self) -> None:
        scenario = validate_scenario_dataframe(make_frame())
        sample = build_focal_sample(scenario, split="train")

        self.assertEqual(sample.scenario_id, SCENARIO_ID)
        self.assertEqual(sample.focal_track_id, "focal-1")
        self.assertEqual(sample.city, "PIT")
        self.assertEqual(sample.history_position.shape, (50, 2))
        self.assertEqual(sample.future_position.shape, (60, 2))
        self.assertEqual(sample.history_position.dtype, np.float32)
        self.assertEqual(sample.history_mask.dtype, np.bool_)
        self.assertTrue(sample.history_mask.all())
        self.assertTrue(sample.future_mask.all())
        self.assertTrue(sample.history_observed.all())
        self.assertFalse(sample.future_observed.any())
        self.assertLess(sample.history_timesteps.max(), sample.future_timesteps.min())
        np.testing.assert_allclose(np.diff(sample.timestamps_ns) / 1e9, 0.12)

    def test_missing_state_is_masked_and_never_interpolated(self) -> None:
        scenario = validate_scenario_dataframe(make_frame(missing_focal_timestep=17))
        sample = build_focal_sample(scenario, split="train")

        self.assertFalse(sample.history_mask[17])
        np.testing.assert_array_equal(sample.history_position[17], np.zeros(2, dtype=np.float32))
        self.assertTrue(sample.history_mask[16])
        self.assertTrue(sample.history_mask[18])

    def test_read_scenario_parquet_uses_validated_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / f"scenario_{SCENARIO_ID}.parquet"
            parquet.touch()
            with patch(
                "trajectory_prediction.data.av2_dataset.pd.read_parquet",
                return_value=make_frame(),
            ) as mocked_read:
                scenario = read_av2_scenario(parquet)
            mocked_read.assert_called_once_with(parquet)
            self.assertEqual(scenario.source_path, parquet)
            self.assertEqual(scenario.scenario_id, SCENARIO_ID)

    def test_rejects_duplicate_timestep(self) -> None:
        frame = make_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ScenarioValidationError, "duplicate timestep"):
            validate_scenario_dataframe(frame)

    def test_rejects_nan_state(self) -> None:
        frame = make_frame()
        frame.loc[0, "position_x"] = np.nan
        with self.assertRaisesRegex(ScenarioValidationError, "NaN or infinite"):
            validate_scenario_dataframe(frame)

    def test_rejects_missing_focal_track(self) -> None:
        frame = make_frame()
        frame["focal_track_id"] = "not-present"
        with self.assertRaisesRegex(ScenarioValidationError, "does not contain focal_track_id"):
            validate_scenario_dataframe(frame)

    def test_rejects_observed_flag_across_history_future_boundary(self) -> None:
        frame = make_frame()
        frame.loc[(frame["track_id"] == "focal-1") & (frame["timestep"] == 50), "observed"] = True
        scenario = validate_scenario_dataframe(frame)
        with self.assertRaisesRegex(ScenarioValidationError, "observed flags"):
            build_focal_sample(scenario, split="train")

    def test_seeded_subset_is_deterministic(self) -> None:
        paths = [Path(f"scene-{index}/scenario_{index:03d}.parquet") for index in range(20)]
        first = select_scenario_paths(paths, 7, 2026)
        second = select_scenario_paths(list(reversed(paths)), 7, 2026)
        different_seed = select_scenario_paths(paths, 7, 2027)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)
        self.assertEqual(first, sorted(first, key=lambda path: path.stem))

    def test_cache_round_trip_preserves_values_and_dtypes(self) -> None:
        sample = build_focal_sample(validate_scenario_dataframe(make_frame()), split="train")
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "cache"
            shards = write_cache_shards([sample], cache_dir, shard_size=1)
            restored = load_cache_shard(cache_dir / shards[0]["file"])[0]
            self.assertEqual(restored.scenario_id, sample.scenario_id)
            self.assertEqual(restored.split, "train")
            np.testing.assert_array_equal(restored.timestamps_ns, sample.timestamps_ns)
            np.testing.assert_array_equal(restored.history_position, sample.history_position)
            np.testing.assert_array_equal(restored.future_mask, sample.future_mask)
            self.assertEqual(restored.history_position.dtype, np.float32)
            self.assertEqual(restored.future_mask.dtype, np.bool_)
            with self.assertRaises(FileExistsError):
                write_cache_shards([sample], cache_dir, shard_size=1)

    def test_fixed_scene_bev_is_written(self) -> None:
        sample = build_focal_sample(
            validate_scenario_dataframe(make_frame(missing_focal_timestep=17)), split="train"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixed_scene.png"
            result = plot_focal_sample_bev(sample, output)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
