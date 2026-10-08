"""Mask-aware geometric metrics for deterministic and multimodal trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np


PerSampleMetrics = dict[str, np.ndarray]
AggregateMetrics = dict[str, float | int]


def _as_numeric_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values; received {array.dtype}.")
    converted = array.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values.")
    return converted


def _normalize_inputs(
    predictions: Any,
    ground_truth: Any,
    future_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predictions)
    target = np.asarray(ground_truth)
    mask = np.asarray(future_mask)

    if predicted.ndim not in {3, 4}:
        raise ValueError(
            "predictions must have rank 3 or 4 with shape [B,T,2] or [B,K,T,2]; "
            f"received {predicted.shape}."
        )
    if predicted.shape[-1] != 2:
        raise ValueError(
            f"predictions must have last dimension 2; received {predicted.shape}."
        )
    if target.ndim != 3:
        raise ValueError(
            "ground_truth must have rank 3 with shape [B,T,2]; "
            f"received {target.shape}."
        )
    if target.shape[-1] != 2:
        raise ValueError(
            f"ground_truth must have last dimension 2; received {target.shape}."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"future_mask must have shape [B,T]; received {mask.shape}."
        )
    if mask.dtype.kind != "b":
        raise ValueError(
            f"future_mask must have boolean dtype; received {mask.dtype}."
        )

    if predicted.ndim == 3:
        batch_size, future_steps, _ = predicted.shape
        mode_count = 1
        predicted = predicted[:, None, :, :]
    else:
        batch_size, mode_count, future_steps, _ = predicted.shape
    if batch_size <= 0 or mode_count <= 0 or future_steps <= 0:
        raise ValueError(
            "predictions batch, mode, and time dimensions must all be positive; "
            f"received B={batch_size}, K={mode_count}, T={future_steps}."
        )
    if target.shape[0] <= 0 or target.shape[1] <= 0:
        raise ValueError(
            "ground_truth batch and time dimensions must be positive; "
            f"received {target.shape}."
        )
    if predicted.shape[0] != target.shape[0] or predicted.shape[2] != target.shape[1]:
        raise ValueError(
            "predictions and ground_truth batch/time dimensions must match; "
            f"received {predicted.shape} and {target.shape}."
        )
    if mask.shape != target.shape[:2]:
        raise ValueError(
            "future_mask must match ground_truth batch/time dimensions; "
            f"received {mask.shape} and {target.shape[:2]}."
        )

    invalid_samples = np.flatnonzero(~mask.any(axis=1)).tolist()
    if invalid_samples:
        raise ValueError(
            "Every sample needs at least one valid future point; "
            f"samples without valid future labels are at indices {invalid_samples}."
        )

    predicted_float = _as_numeric_array(predicted, name="predictions")
    target_float = _as_numeric_array(target, name="ground_truth")
    return predicted_float, target_float, mask


def _validate_miss_threshold(miss_threshold_m: Any) -> float:
    threshold = np.asarray(miss_threshold_m)
    if threshold.shape != () or threshold.dtype.kind not in "iuf":
        raise ValueError("miss_threshold_m must be one finite positive scalar in metres.")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value <= 0.0:
        raise ValueError(
            "miss_threshold_m must be one finite positive scalar in metres; "
            f"received {miss_threshold_m!r}."
        )
    return threshold_value


def compute_ade_fde(
    predictions: Any,
    ground_truth: Any,
    future_mask: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-mode ADE/FDE arrays ``[B,K]`` as float64 metres.

    ``predictions`` accepts deterministic ``[B,T,2]`` or multimodal
    ``[B,K,T,2]`` trajectories. ``ground_truth`` is ``[B,T,2]`` and
    ``future_mask`` is boolean ``[B,T]``. ADE averages only valid future
    points; FDE uses each sample's final valid future point. The returned tuple
    is ``(ade_by_mode, fde_by_mode)`` and never modifies an input array.
    """

    predicted, target, mask = _normalize_inputs(predictions, ground_truth, future_mask)
    displacements = np.linalg.norm(predicted - target[:, None, :, :], axis=-1)
    valid_counts = mask.sum(axis=1, dtype=np.int64)
    ade_by_mode = (displacements * mask[:, None, :]).sum(axis=2) / valid_counts[:, None]

    batch_size, mode_count, future_steps = displacements.shape
    last_valid = future_steps - 1 - np.argmax(mask[:, ::-1], axis=1)
    batch_indices = np.arange(batch_size)[:, None]
    mode_indices = np.arange(mode_count)[None, :]
    fde_by_mode = displacements[batch_indices, mode_indices, last_valid[:, None]]
    return ade_by_mode.astype(np.float64, copy=False), fde_by_mode.astype(
        np.float64, copy=False
    )


def compute_per_sample_forecasting_metrics(
    predictions: Any,
    ground_truth: Any,
    future_mask: Any,
    *,
    miss_threshold_m: float = 2.0,
) -> PerSampleMetrics:
    """Return mask-aware per-sample geometric metrics in metres.

    ``predictions`` is ``[B,T,2]`` or ``[B,K,T,2]``; ``ground_truth`` and the
    boolean ``future_mask`` are ``[B,T,2]`` and ``[B,T]``. The result contains
    float64 ``ade_by_mode``/``fde_by_mode`` arrays ``[B,K]``, float64
    ``min_ade``/``min_fde`` arrays ``[B]``, and boolean ``miss`` ``[B]``.
    Invalid masked timesteps do not contribute. minADE and minFDE independently
    choose their best mode. ``miss`` is true only when minFDE is strictly
    greater than the finite positive ``miss_threshold_m`` (default 2 metres).
    """

    threshold = _validate_miss_threshold(miss_threshold_m)
    ade_by_mode, fde_by_mode = compute_ade_fde(
        predictions, ground_truth, future_mask
    )
    min_ade = np.min(ade_by_mode, axis=1)
    min_fde = np.min(fde_by_mode, axis=1)
    return {
        "ade_by_mode": ade_by_mode,
        "fde_by_mode": fde_by_mode,
        "min_ade": min_ade,
        "min_fde": min_fde,
        "miss": min_fde > threshold,
    }


def compute_forecasting_metrics(
    predictions: Any,
    ground_truth: Any,
    future_mask: Any,
    *,
    miss_threshold_m: float = 2.0,
) -> AggregateMetrics:
    """Aggregate mask-aware trajectory metrics by averaging samples.

    ``predictions`` is ``[B,T,2]`` or ``[B,K,T,2]``; ``ground_truth`` and the
    boolean ``future_mask`` are ``[B,T,2]`` and ``[B,T]``. Metrics are computed
    per sample using only valid timesteps, then averaged without frame-count
    weighting. The result always contains metre-valued ``minADE``/``minFDE``,
    ``MissRate``, sample/mode counts, and the finite positive miss threshold in
    metres. minADE and minFDE select modes independently, and a miss requires
    minFDE to be strictly greater than the threshold. K=1 inputs also include
    ``ADE`` and ``FDE``; K>1 inputs omit those ambiguous keys.
    """

    threshold = _validate_miss_threshold(miss_threshold_m)
    per_sample = compute_per_sample_forecasting_metrics(
        predictions,
        ground_truth,
        future_mask,
        miss_threshold_m=threshold,
    )
    sample_count, mode_count = per_sample["ade_by_mode"].shape
    result: AggregateMetrics = {
        "minADE": float(np.mean(per_sample["min_ade"], dtype=np.float64)),
        "minFDE": float(np.mean(per_sample["min_fde"], dtype=np.float64)),
        "MissRate": float(np.mean(per_sample["miss"], dtype=np.float64)),
        "sample_count": int(sample_count),
        "mode_count": int(mode_count),
        "miss_threshold_m": threshold,
    }
    if mode_count == 1:
        result["ADE"] = float(
            np.mean(per_sample["ade_by_mode"][:, 0], dtype=np.float64)
        )
        result["FDE"] = float(
            np.mean(per_sample["fde_by_mode"][:, 0], dtype=np.float64)
        )
    return result
