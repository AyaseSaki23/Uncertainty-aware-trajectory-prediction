"""Hand-checked tests for mask-aware deterministic and multimodal metrics."""

from __future__ import annotations

import unittest

import numpy as np

from trajectory_prediction.metrics.forecasting import (
    compute_ade_fde,
    compute_forecasting_metrics,
    compute_per_sample_forecasting_metrics,
)


class ForecastingMetricsTest(unittest.TestCase):
    def test_zero_error_and_k1_metrics_are_consistent(self) -> None:
        ground_truth = np.asarray(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32
        )
        mask = np.asarray([[True, True, True]])

        ade, fde = compute_ade_fde(ground_truth.copy(), ground_truth, mask)
        per_sample = compute_per_sample_forecasting_metrics(
            ground_truth.copy(), ground_truth, mask
        )
        aggregate = compute_forecasting_metrics(ground_truth.copy(), ground_truth, mask)

        np.testing.assert_array_equal(ade, [[0.0]])
        np.testing.assert_array_equal(fde, [[0.0]])
        np.testing.assert_array_equal(per_sample["min_ade"], ade[:, 0])
        np.testing.assert_array_equal(per_sample["min_fde"], fde[:, 0])
        self.assertEqual(aggregate["ADE"], 0.0)
        self.assertEqual(aggregate["FDE"], 0.0)
        self.assertEqual(aggregate["minADE"], 0.0)
        self.assertEqual(aggregate["minFDE"], 0.0)
        self.assertEqual(aggregate["MissRate"], 0.0)

    def test_fixed_three_four_translation_has_five_metre_error(self) -> None:
        ground_truth = np.asarray(
            [[[0.0, 0.0], [1.0, 2.0], [-2.0, 3.0]]], dtype=np.float64
        )
        predictions = ground_truth + np.asarray([3.0, 4.0])
        mask = np.asarray([[True, True, True]])

        ade, fde = compute_ade_fde(predictions, ground_truth, mask)
        np.testing.assert_allclose(ade, [[5.0]])
        np.testing.assert_allclose(fde, [[5.0]])

    def test_different_valid_lengths_are_computed_per_sample(self) -> None:
        ground_truth = np.zeros((2, 3, 2), dtype=np.float64)
        predictions = np.asarray(
            [
                [[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
                [[2.0, 0.0], [100.0, 0.0], [4.0, 0.0]],
            ]
        )
        mask = np.asarray([[True, True, False], [True, False, True]])

        ade, fde = compute_ade_fde(predictions, ground_truth, mask)
        np.testing.assert_allclose(ade[:, 0], [2.0, 3.0])
        np.testing.assert_allclose(fde[:, 0], [3.0, 4.0])

    def test_fde_uses_last_valid_point_and_ignores_missing_placeholders(self) -> None:
        ground_truth = np.zeros((1, 4, 2), dtype=np.float64)
        predictions = np.asarray(
            [[[1.0, 0.0], [50.0, 0.0], [3.0, 0.0], [100.0, 0.0]]]
        )
        mask = np.asarray([[True, False, True, False]])

        ade, fde = compute_ade_fde(predictions, ground_truth, mask)
        np.testing.assert_allclose(ade, [[2.0]])
        np.testing.assert_allclose(fde, [[3.0]])

    def test_deterministic_and_explicit_single_mode_inputs_match(self) -> None:
        ground_truth = np.zeros((2, 2, 2), dtype=np.float64)
        deterministic = np.asarray(
            [[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]]
        )
        mask = np.ones((2, 2), dtype=bool)

        deterministic_result = compute_per_sample_forecasting_metrics(
            deterministic, ground_truth, mask
        )
        multimodal_result = compute_per_sample_forecasting_metrics(
            deterministic[:, None, :, :], ground_truth, mask
        )
        for key in deterministic_result:
            np.testing.assert_array_equal(deterministic_result[key], multimodal_result[key])

    def test_exact_and_repeated_modes_have_stable_minimum(self) -> None:
        ground_truth = np.asarray([[[0.0, 0.0], [1.0, 0.0]]])
        offset_mode = ground_truth + np.asarray([3.0, 4.0])
        predictions = np.stack(
            [offset_mode, ground_truth, ground_truth], axis=1
        )
        mask = np.asarray([[True, True]])

        result = compute_per_sample_forecasting_metrics(
            predictions, ground_truth, mask
        )
        np.testing.assert_allclose(result["ade_by_mode"], [[5.0, 0.0, 0.0]])
        np.testing.assert_allclose(result["fde_by_mode"], [[5.0, 0.0, 0.0]])
        np.testing.assert_allclose(result["min_ade"], [0.0])
        np.testing.assert_allclose(result["min_fde"], [0.0])

    def test_min_ade_and_min_fde_select_modes_independently(self) -> None:
        ground_truth = np.zeros((1, 2, 2), dtype=np.float64)
        predictions = np.asarray(
            [[[[0.0, 0.0], [4.0, 0.0]], [[3.0, 0.0], [3.0, 0.0]]]]
        )
        mask = np.asarray([[True, True]])

        result = compute_per_sample_forecasting_metrics(
            predictions, ground_truth, mask
        )
        np.testing.assert_allclose(result["ade_by_mode"], [[2.0, 3.0]])
        np.testing.assert_allclose(result["fde_by_mode"], [[4.0, 3.0]])
        np.testing.assert_allclose(result["min_ade"], [2.0])
        np.testing.assert_allclose(result["min_fde"], [3.0])

    def test_miss_rate_uses_strictly_greater_than_threshold(self) -> None:
        ground_truth = np.zeros((2, 1, 2), dtype=np.float64)
        predictions = np.asarray([[[2.0, 0.0]], [[2.0001, 0.0]]])
        mask = np.asarray([[True], [True]])

        per_sample = compute_per_sample_forecasting_metrics(
            predictions, ground_truth, mask, miss_threshold_m=2.0
        )
        aggregate = compute_forecasting_metrics(
            predictions, ground_truth, mask, miss_threshold_m=2.0
        )
        np.testing.assert_array_equal(per_sample["miss"], [False, True])
        self.assertEqual(aggregate["MissRate"], 0.5)
        self.assertEqual(aggregate["miss_threshold_m"], 2.0)

    def test_aggregate_averages_samples_not_all_valid_frames(self) -> None:
        ground_truth = np.zeros((2, 3, 2), dtype=np.float64)
        predictions = np.asarray(
            [
                [[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        mask = np.asarray([[True, False, False], [True, True, True]])

        aggregate = compute_forecasting_metrics(predictions, ground_truth, mask)
        self.assertEqual(aggregate["ADE"], 5.0)
        self.assertEqual(aggregate["minADE"], 5.0)
        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["mode_count"], 1)

    def test_multimodal_aggregate_omits_ambiguous_ade_fde_keys(self) -> None:
        ground_truth = np.zeros((1, 2, 2), dtype=np.float64)
        predictions = np.zeros((1, 2, 2, 2), dtype=np.float64)
        mask = np.asarray([[True, True]])

        aggregate = compute_forecasting_metrics(predictions, ground_truth, mask)
        self.assertNotIn("ADE", aggregate)
        self.assertNotIn("FDE", aggregate)
        self.assertEqual(aggregate["mode_count"], 2)

    def test_float_inputs_return_float64_and_inputs_are_not_mutated(self) -> None:
        for dtype in (np.float32, np.float64):
            ground_truth = np.zeros((1, 2, 2), dtype=dtype)
            predictions = np.ones((1, 2, 2), dtype=dtype)
            mask = np.asarray([[True, True]])
            predictions_before = predictions.copy()
            ground_truth_before = ground_truth.copy()
            mask_before = mask.copy()

            ade, fde = compute_ade_fde(predictions, ground_truth, mask)
            self.assertEqual(ade.dtype, np.dtype(np.float64))
            self.assertEqual(fde.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(predictions, predictions_before)
            np.testing.assert_array_equal(ground_truth, ground_truth_before)
            np.testing.assert_array_equal(mask, mask_before)

    def test_rejects_prediction_rank_and_coordinate_dimension(self) -> None:
        ground_truth = np.zeros((1, 2, 2))
        mask = np.ones((1, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "rank 3 or 4"):
            compute_ade_fde(np.zeros((2, 2)), ground_truth, mask)
        with self.assertRaisesRegex(ValueError, "last dimension 2"):
            compute_ade_fde(np.zeros((1, 2, 3)), ground_truth, mask)
        with self.assertRaisesRegex(ValueError, "last dimension 2"):
            compute_ade_fde(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)), mask)

    def test_rejects_batch_time_and_mask_mismatches(self) -> None:
        predictions = np.zeros((2, 3, 2))
        ground_truth = np.zeros((2, 3, 2))
        mask = np.ones((2, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "batch/time"):
            compute_ade_fde(predictions[:1], ground_truth, mask)
        with self.assertRaisesRegex(ValueError, "batch/time"):
            compute_ade_fde(predictions, ground_truth[:, :2], mask)
        with self.assertRaisesRegex(ValueError, "future_mask"):
            compute_ade_fde(predictions, ground_truth, mask[:, :2])
        with self.assertRaisesRegex(ValueError, "boolean"):
            compute_ade_fde(predictions, ground_truth, mask.astype(np.int64))

    def test_rejects_samples_without_valid_future_and_lists_indices(self) -> None:
        predictions = np.zeros((2, 2, 2))
        ground_truth = np.zeros((2, 2, 2))
        mask = np.asarray([[True, False], [False, False]])
        with self.assertRaisesRegex(ValueError, r"indices \[1\]"):
            compute_ade_fde(predictions, ground_truth, mask)

    def test_rejects_nonfinite_values(self) -> None:
        predictions = np.zeros((1, 2, 2))
        ground_truth = np.zeros((1, 2, 2))
        mask = np.ones((1, 2), dtype=bool)
        predictions[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "predictions.*finite"):
            compute_ade_fde(predictions, ground_truth, mask)
        predictions[0, 0, 0] = 0.0
        ground_truth[0, 1, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "ground_truth.*finite"):
            compute_ade_fde(predictions, ground_truth, mask)

    def test_rejects_empty_batch_mode_or_time_dimensions(self) -> None:
        cases = [
            (np.zeros((0, 2, 2)), np.zeros((0, 2, 2)), np.zeros((0, 2), dtype=bool)),
            (np.zeros((1, 0, 2, 2)), np.zeros((1, 2, 2)), np.ones((1, 2), dtype=bool)),
            (np.zeros((1, 0, 2)), np.zeros((1, 0, 2)), np.zeros((1, 0), dtype=bool)),
        ]
        for predictions, ground_truth, mask in cases:
            with self.subTest(shape=predictions.shape):
                with self.assertRaisesRegex(ValueError, "positive"):
                    compute_ade_fde(predictions, ground_truth, mask)

    def test_rejects_invalid_miss_thresholds(self) -> None:
        predictions = np.zeros((1, 1, 2))
        ground_truth = np.zeros((1, 1, 2))
        mask = np.ones((1, 1), dtype=bool)
        for threshold in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "miss_threshold_m"):
                    compute_per_sample_forecasting_metrics(
                        predictions,
                        ground_truth,
                        mask,
                        miss_threshold_m=threshold,
                    )


if __name__ == "__main__":
    unittest.main()
