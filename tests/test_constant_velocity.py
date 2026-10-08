"""Tests for the strictly causal constant-velocity forecasting baseline."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.metrics.forecasting import compute_forecasting_metrics
from trajectory_prediction.models.constant_velocity import (
    predict_constant_velocity,
    predict_constant_velocity_sample,
)


def make_cv_sample() -> TrajectorySample:
    """Create a local-frame sample whose labels follow constant velocity."""

    return TrajectorySample(
        scenario_id="cv-test",
        split="val",
        focal_track_id="focal",
        city="PIT",
        history_timesteps=np.arange(4, dtype=np.int64),
        future_timesteps=np.arange(4, 7, dtype=np.int64),
        timestamps_ns=np.asarray(
            [0, 100_000_000, 250_000_000, 400_000_000, 500_000_000,
             700_000_000, 1_000_000_000],
            dtype=np.int64,
        ),
        history_position=np.asarray(
            [[-0.8, 0.4], [0.0, 0.0], [-0.3, 0.15], [0.0, 0.0]],
            dtype=np.float32,
        ),
        history_velocity=np.asarray(
            [[2.0, -1.0], [0.0, 0.0], [2.0, -1.0], [2.0, -1.0]],
            dtype=np.float32,
        ),
        history_heading=np.zeros(4, dtype=np.float32),
        history_observed=np.asarray([True, False, True, True]),
        history_mask=np.asarray([True, False, True, True]),
        future_position=np.asarray(
            [[0.2, -0.1], [0.6, -0.3], [1.2, -0.6]], dtype=np.float32
        ),
        future_velocity=np.tile(
            np.asarray([2.0, -1.0], dtype=np.float32), (3, 1)
        ),
        future_heading=np.zeros(3, dtype=np.float32),
        future_observed=np.zeros(3, dtype=bool),
        future_mask=np.ones(3, dtype=bool),
        origin_xy=np.asarray([10.0, 20.0], dtype=np.float32),
        reference_heading_rad=0.5,
        coordinate_frame="local",
    )


class ConstantVelocityTest(unittest.TestCase):
    def test_stationary_single_trajectory(self) -> None:
        history_position = np.asarray([[3.0, -2.0], [3.0, -2.0]])
        history_velocity = np.zeros((2, 2))
        history_mask = np.asarray([True, True])
        prediction = predict_constant_velocity(
            history_position,
            history_velocity,
            history_mask,
            np.asarray([0, 500_000_000]),
            np.asarray([750_000_000, 1_500_000_000]),
        )

        np.testing.assert_array_equal(prediction, [[3.0, -2.0], [3.0, -2.0]])

    def test_straight_line_prediction_has_zero_ade_and_fde(self) -> None:
        prediction = predict_constant_velocity(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            np.asarray([True, True]),
            np.asarray([0, 1_000_000_000]),
            np.asarray([2_000_000_000, 3_000_000_000]),
        )
        ground_truth = np.asarray([[[2.0, 0.0], [3.0, 0.0]]])
        metrics = compute_forecasting_metrics(
            prediction[None, :, :], ground_truth, np.ones((1, 2), dtype=bool)
        )

        np.testing.assert_allclose(prediction, [[2.0, 0.0], [3.0, 0.0]])
        self.assertEqual(metrics["ADE"], 0.0)
        self.assertEqual(metrics["FDE"], 0.0)

    def test_nonuniform_future_intervals_use_real_timestamps(self) -> None:
        prediction = predict_constant_velocity(
            np.asarray([[4.0, 5.0]]),
            np.asarray([[2.0, -4.0]]),
            np.asarray([True]),
            np.asarray([1_000_000_000]),
            np.asarray([1_100_000_000, 1_350_000_000, 2_000_000_000]),
        )

        np.testing.assert_allclose(
            prediction,
            [[4.2, 4.6], [4.7, 3.6], [6.0, 1.0]],
            atol=1e-12,
        )

    def test_trailing_missing_history_uses_last_valid_state_and_time(self) -> None:
        prediction = predict_constant_velocity(
            np.asarray([[0.0, 0.0], [2.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            np.asarray([[1.0, 0.0], [2.0, -1.0], [0.0, 0.0], [0.0, 0.0]]),
            np.asarray([True, True, False, False]),
            np.asarray([0, 1_000_000_000, 2_000_000_000, 3_000_000_000]),
            np.asarray([4_000_000_000, 4_500_000_000]),
        )

        np.testing.assert_allclose(prediction, [[8.0, -2.0], [9.0, -2.5]])

    def test_batch_uses_each_samples_last_valid_history(self) -> None:
        prediction = predict_constant_velocity(
            np.asarray(
                [
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                    [[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                    [[0.0, -2.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            np.asarray([[True, True, True], [True, False, False]]),
            np.asarray(
                [[0, 1_000_000_000, 2_000_000_000], [0, 1_000_000_000, 2_000_000_000]]
            ),
            np.asarray(
                [[3_000_000_000, 4_000_000_000], [3_000_000_000, 4_000_000_000]]
            ),
        )

        self.assertEqual(prediction.shape, (2, 2, 2))
        np.testing.assert_allclose(prediction[0], [[3.0, 0.0], [4.0, 0.0]])
        np.testing.assert_allclose(prediction[1], [[5.0, -1.0], [5.0, -3.0]])

    def test_sample_interface_uses_timestep_indices_and_same_frame(self) -> None:
        sample = make_cv_sample()
        prediction = predict_constant_velocity_sample(sample)

        self.assertEqual(sample.coordinate_frame, "local")
        self.assertEqual(prediction.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(prediction, sample.future_position, atol=1e-7)

    def test_future_label_changes_do_not_change_sample_prediction(self) -> None:
        sample = make_cv_sample()
        changed = replace(
            sample,
            future_position=sample.future_position + np.float32(1000.0),
            future_velocity=sample.future_velocity - np.float32(500.0),
            future_heading=sample.future_heading + np.float32(2.0),
            future_mask=np.asarray([False, True, False]),
        )

        original_prediction = predict_constant_velocity_sample(sample)
        changed_prediction = predict_constant_velocity_sample(changed)
        np.testing.assert_array_equal(changed_prediction, original_prediction)

    def test_inputs_are_not_modified_and_output_is_float64(self) -> None:
        for dtype in (np.float32, np.float64):
            positions = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)
            velocities = np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=dtype)
            mask = np.asarray([True, True])
            history_time = np.asarray([0, 1_000_000_000], dtype=np.int64)
            future_time = np.asarray([2_000_000_000], dtype=np.int64)
            originals = tuple(
                value.copy()
                for value in (positions, velocities, mask, history_time, future_time)
            )

            prediction = predict_constant_velocity(
                positions, velocities, mask, history_time, future_time
            )

            self.assertEqual(prediction.dtype, np.dtype(np.float64))
            for value, original in zip(
                (positions, velocities, mask, history_time, future_time), originals
            ):
                np.testing.assert_array_equal(value, original)

    def test_rejects_invalid_position_velocity_and_mask_contracts(self) -> None:
        valid_position = np.zeros((2, 2))
        valid_velocity = np.zeros((2, 2))
        valid_mask = np.ones(2, dtype=bool)
        history_time = np.asarray([0, 1])
        future_time = np.asarray([2])
        with self.assertRaisesRegex(ValueError, "history_position"):
            predict_constant_velocity(
                np.zeros((2, 3)), valid_velocity, valid_mask, history_time, future_time
            )
        with self.assertRaisesRegex(ValueError, "history_velocity"):
            predict_constant_velocity(
                valid_position, np.zeros((3, 2)), valid_mask, history_time, future_time
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            predict_constant_velocity(
                valid_position,
                valid_velocity,
                valid_mask.astype(np.int64),
                history_time,
                future_time,
            )

    def test_rejects_samples_without_valid_history_and_lists_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, r"indices \[1\]"):
            predict_constant_velocity(
                np.zeros((2, 2, 2)),
                np.zeros((2, 2, 2)),
                np.asarray([[True, False], [False, False]]),
                np.asarray([[0, 1], [0, 1]]),
                np.asarray([[2], [2]]),
            )

    def test_rejects_nonfinite_states_or_timestamps(self) -> None:
        position = np.zeros((1, 2))
        velocity = np.zeros((1, 2))
        mask = np.asarray([True])
        with self.assertRaisesRegex(ValueError, "history_position.*finite"):
            predict_constant_velocity(
                position + np.nan, velocity, mask, np.asarray([0]), np.asarray([1])
            )
        with self.assertRaisesRegex(ValueError, "history_velocity.*finite"):
            predict_constant_velocity(
                position, velocity + np.inf, mask, np.asarray([0]), np.asarray([1])
            )
        with self.assertRaisesRegex(ValueError, "future_timestamps_ns.*finite"):
            predict_constant_velocity(
                position, velocity, mask, np.asarray([0]), np.asarray([np.nan])
            )

    def test_rejects_nonmonotonic_or_noncausal_timestamps(self) -> None:
        position = np.zeros((2, 2))
        velocity = np.zeros((2, 2))
        mask = np.asarray([True, True])
        with self.assertRaisesRegex(ValueError, "history_timestamps_ns.*increasing"):
            predict_constant_velocity(
                position, velocity, mask, np.asarray([1, 1]), np.asarray([2])
            )
        with self.assertRaisesRegex(ValueError, "future_timestamps_ns.*increasing"):
            predict_constant_velocity(
                position, velocity, mask, np.asarray([0, 1]), np.asarray([3, 2])
            )
        with self.assertRaisesRegex(ValueError, "after the history window"):
            predict_constant_velocity(
                position, velocity, mask, np.asarray([0, 2]), np.asarray([1])
            )

    def test_rejects_empty_dimensions_and_timestamp_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            predict_constant_velocity(
                np.zeros((0, 2)),
                np.zeros((0, 2)),
                np.zeros(0, dtype=bool),
                np.zeros(0),
                np.asarray([1]),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            predict_constant_velocity(
                np.zeros((1, 2)),
                np.zeros((1, 2)),
                np.ones(1, dtype=bool),
                np.asarray([0]),
                np.zeros(0),
            )
        with self.assertRaisesRegex(ValueError, "future_timestamps_ns"):
            predict_constant_velocity(
                np.zeros((2, 2, 2)),
                np.zeros((2, 2, 2)),
                np.ones((2, 2), dtype=bool),
                np.asarray([[0, 1], [0, 1]]),
                np.asarray([2, 3]),
            )

    def test_sample_interface_rejects_invalid_timestep_indices(self) -> None:
        sample = make_cv_sample()
        with self.assertRaisesRegex(ValueError, "integer indices"):
            predict_constant_velocity_sample(
                replace(sample, future_timesteps=np.asarray([4.0, 5.0, 6.0]))
            )
        with self.assertRaisesRegex(ValueError, "outside timestamps_ns"):
            predict_constant_velocity_sample(
                replace(sample, future_timesteps=np.asarray([4, 5, 99]))
            )


if __name__ == "__main__":
    unittest.main()
