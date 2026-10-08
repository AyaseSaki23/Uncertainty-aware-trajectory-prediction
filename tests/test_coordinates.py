"""Tests for causal global/target-local coordinate transformations."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.data.av2_dataset import load_cache_shard, write_cache_shards
from trajectory_prediction.geometry.coordinates import (
    global_heading_to_local,
    global_to_local,
    global_vector_to_local,
    local_heading_to_global,
    local_to_global,
    local_vector_to_global,
    trajectory_sample_to_global,
    trajectory_sample_to_local,
    wrap_angle,
)


def make_sample() -> TrajectorySample:
    """Create a small global-frame sample with masked history and future slots."""

    return TrajectorySample(
        scenario_id="coordinate-test",
        split="train",
        focal_track_id="focal-1",
        city="PIT",
        history_timesteps=np.arange(4, dtype=np.int64),
        future_timesteps=np.arange(4, 7, dtype=np.int64),
        timestamps_ns=np.arange(7, dtype=np.int64) * 100_000_000,
        history_position=np.asarray(
            [[8.0, 19.0], [0.0, 0.0], [9.0, 20.0], [10.0, 20.0]], dtype=np.float32
        ),
        history_velocity=np.asarray(
            [[0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
        ),
        history_heading=np.asarray(
            [np.pi / 2, 0.0, np.pi / 2, np.pi / 2], dtype=np.float32
        ),
        history_observed=np.asarray([True, False, True, True]),
        history_mask=np.asarray([True, False, True, True]),
        future_position=np.asarray(
            [[10.0, 21.0], [0.0, 0.0], [9.0, 20.0]], dtype=np.float32
        ),
        future_velocity=np.asarray(
            [[0.0, 1.0], [0.0, 0.0], [-1.0, 0.0]], dtype=np.float32
        ),
        future_heading=np.asarray(
            [np.pi / 2, 0.0, np.pi], dtype=np.float32
        ),
        future_observed=np.asarray([False, False, False]),
        future_mask=np.asarray([True, False, True]),
        origin_xy=np.asarray([10.0, 20.0], dtype=np.float32),
        reference_heading_rad=float(np.float32(np.pi / 2)),
    )


class CoordinateTransformTest(unittest.TestCase):
    def test_zero_heading_and_nonzero_translation(self) -> None:
        points = np.asarray([[3.0, 5.0], [4.0, 7.0]], dtype=np.float64)
        local = global_to_local(points, np.asarray([3.0, 5.0]), 0.0)
        np.testing.assert_allclose(local, [[0.0, 0.0], [1.0, 2.0]])
        np.testing.assert_allclose(local_to_global(local, [3.0, 5.0], 0.0), points)

    def test_ninety_degree_reference_maps_global_y_to_local_x(self) -> None:
        point = np.asarray([[0.0, 1.0]], dtype=np.float64)
        local = global_to_local(point, [0.0, 0.0], np.pi / 2)
        np.testing.assert_allclose(local, [[1.0, 0.0]], atol=1e-12)

    def test_random_position_round_trip_for_float32_and_float64(self) -> None:
        generator = np.random.default_rng(2026)
        for dtype, tolerance in ((np.float32, 1e-4), (np.float64, 1e-11)):
            points = generator.normal(size=(3, 5, 2)).astype(dtype)
            original = points.copy()
            origin = np.asarray([123.25, -57.5], dtype=dtype)
            local = global_to_local(points, origin, 1.234)
            restored = local_to_global(local, origin, 1.234)
            self.assertEqual(local.dtype, np.dtype(dtype))
            self.assertEqual(restored.dtype, np.dtype(dtype))
            self.assertLessEqual(float(np.max(np.abs(restored - points))), tolerance)
            np.testing.assert_array_equal(points, original)

    def test_random_vector_round_trip_and_arbitrary_leading_dimensions(self) -> None:
        generator = np.random.default_rng(17)
        vectors = generator.normal(size=(2, 3, 4, 2)).astype(np.float32)
        original = vectors.copy()
        local = global_vector_to_local(vectors, -0.73)
        restored = local_vector_to_global(local, -0.73)
        self.assertEqual(local.shape, vectors.shape)
        np.testing.assert_allclose(restored, vectors, atol=1e-6)
        np.testing.assert_array_equal(vectors, original)

    def test_heading_wrap_and_circular_round_trip(self) -> None:
        angles = np.asarray(
            [np.pi, -np.pi, 3 * np.pi, -3 * np.pi, np.pi + 0.1, -np.pi - 0.1]
        )
        expected = np.asarray([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi + 0.1, np.pi - 0.1])
        np.testing.assert_allclose(wrap_angle(angles), expected, atol=1e-12)

        headings = np.asarray([np.pi - 1e-8, -np.pi + 1e-8, 0.25], dtype=np.float64)
        local = global_heading_to_local(headings, np.pi - 0.2)
        restored = local_heading_to_global(local, np.pi - 0.2)
        np.testing.assert_allclose(wrap_angle(restored - headings), 0.0, atol=1e-12)

    def test_rejects_invalid_shapes_and_nonfinite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "last dimension"):
            global_to_local(np.zeros(3), [0.0, 0.0], 0.0)
        with self.assertRaisesRegex(ValueError, "origin_xy"):
            global_to_local(np.zeros((2, 2)), [0.0, 0.0, 0.0], 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            global_to_local(np.asarray([[np.inf, 0.0]]), [0.0, 0.0], 0.0)
        with self.assertRaisesRegex(ValueError, "reference_heading_rad"):
            global_vector_to_local(np.zeros((2, 2)), np.nan)

    def test_sample_conversion_preserves_masks_and_zero_placeholders(self) -> None:
        sample = make_sample()
        local = trajectory_sample_to_local(sample)

        self.assertEqual(sample.coordinate_frame, "global")
        self.assertEqual(local.coordinate_frame, "local")
        np.testing.assert_array_equal(local.history_mask, sample.history_mask)
        np.testing.assert_array_equal(local.future_mask, sample.future_mask)
        np.testing.assert_array_equal(local.history_observed, sample.history_observed)
        np.testing.assert_array_equal(local.future_observed, sample.future_observed)
        np.testing.assert_array_equal(local.history_timesteps, sample.history_timesteps)
        np.testing.assert_array_equal(local.future_timesteps, sample.future_timesteps)
        np.testing.assert_array_equal(local.timestamps_ns, sample.timestamps_ns)
        np.testing.assert_array_equal(local.history_position[~local.history_mask], 0.0)
        np.testing.assert_array_equal(local.history_velocity[~local.history_mask], 0.0)
        np.testing.assert_array_equal(local.history_heading[~local.history_mask], 0.0)
        np.testing.assert_array_equal(local.future_position[~local.future_mask], 0.0)
        np.testing.assert_array_equal(local.future_velocity[~local.future_mask], 0.0)
        np.testing.assert_array_equal(local.future_heading[~local.future_mask], 0.0)

        last_valid = int(np.flatnonzero(local.history_mask)[-1])
        np.testing.assert_allclose(local.history_position[last_valid], [0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(local.history_heading[last_valid]), 0.0, places=6)
        np.testing.assert_allclose(local.future_position[0], [1.0, 0.0], atol=1e-6)

    def test_sample_round_trip_restores_valid_states_without_mutating_input(self) -> None:
        sample = make_sample()
        history_before = sample.history_position.copy()
        future_before = sample.future_position.copy()
        restored = trajectory_sample_to_global(trajectory_sample_to_local(sample))

        self.assertEqual(restored.coordinate_frame, "global")
        np.testing.assert_allclose(
            restored.history_position[sample.history_mask],
            sample.history_position[sample.history_mask],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            restored.future_position[sample.future_mask],
            sample.future_position[sample.future_mask],
            atol=1e-5,
        )
        np.testing.assert_array_equal(restored.history_position[~sample.history_mask], 0.0)
        np.testing.assert_array_equal(restored.future_position[~sample.future_mask], 0.0)
        np.testing.assert_array_equal(sample.history_position, history_before)
        np.testing.assert_array_equal(sample.future_position, future_before)

    def test_future_changes_do_not_change_reference_or_local_history(self) -> None:
        sample = make_sample()
        changed = replace(
            sample,
            future_position=sample.future_position + np.float32(1000.0),
            future_velocity=sample.future_velocity - np.float32(500.0),
            future_heading=sample.future_heading + np.float32(1.0),
        )
        original_local = trajectory_sample_to_local(sample)
        changed_local = trajectory_sample_to_local(changed)

        np.testing.assert_array_equal(changed_local.origin_xy, original_local.origin_xy)
        self.assertEqual(
            changed_local.reference_heading_rad, original_local.reference_heading_rad
        )
        np.testing.assert_array_equal(
            changed_local.history_position, original_local.history_position
        )
        np.testing.assert_array_equal(
            changed_local.history_velocity, original_local.history_velocity
        )
        np.testing.assert_array_equal(
            changed_local.history_heading, original_local.history_heading
        )

    def test_sample_conversion_rejects_invalid_history_or_reference_metadata(self) -> None:
        sample = make_sample()
        with self.assertRaisesRegex(ValueError, "no valid historical state"):
            trajectory_sample_to_local(
                replace(sample, history_mask=np.zeros_like(sample.history_mask))
            )
        with self.assertRaisesRegex(ValueError, "origin_xy does not match"):
            trajectory_sample_to_local(
                replace(sample, origin_xy=np.asarray([999.0, 999.0], dtype=np.float32))
            )
        with self.assertRaisesRegex(ValueError, "coordinate_frame"):
            replace(sample, coordinate_frame="map")

    def test_legacy_cache_without_coordinate_frame_defaults_to_global(self) -> None:
        sample = make_sample()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = write_cache_shards([sample], root / "new-cache", shard_size=1)
            new_shard = root / "new-cache" / shards[0]["file"]
            with np.load(new_shard, allow_pickle=False) as payload:
                legacy_payload = {
                    key: payload[key].copy()
                    for key in payload.files
                    if key != "coordinate_frame"
                }
            legacy_shard = root / "legacy-shard.npz"
            np.savez_compressed(legacy_shard, **legacy_payload)

            restored = load_cache_shard(legacy_shard)[0]
            self.assertEqual(restored.coordinate_frame, "global")

    def test_new_cache_preserves_explicit_local_coordinate_frame(self) -> None:
        local = trajectory_sample_to_local(make_sample())
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "local-cache"
            shards = write_cache_shards([local], cache_dir, shard_size=1)
            restored = load_cache_shard(cache_dir / shards[0]["file"])[0]

            self.assertEqual(restored.coordinate_frame, "local")
            np.testing.assert_array_equal(restored.history_position, local.history_position)


if __name__ == "__main__":
    unittest.main()
