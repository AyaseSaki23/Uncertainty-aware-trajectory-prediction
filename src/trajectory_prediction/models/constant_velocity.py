"""Strictly causal deterministic constant-velocity forecasting baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from trajectory_prediction.contracts import TrajectorySample


NANOSECONDS_PER_SECOND = 1_000_000_000.0


def _as_float64_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values; received {array.dtype}.")
    converted = array.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values.")
    return converted


def _normalize_cv_inputs(
    history_position: Any,
    history_velocity: Any,
    history_mask: Any,
    history_timestamps_ns: Any,
    future_timestamps_ns: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    positions_raw = np.asarray(history_position)
    velocities_raw = np.asarray(history_velocity)
    mask = np.asarray(history_mask)
    history_time_raw = np.asarray(history_timestamps_ns)
    future_time_raw = np.asarray(future_timestamps_ns)

    if positions_raw.ndim not in {2, 3} or positions_raw.shape[-1:] != (2,):
        raise ValueError(
            "history_position must have shape [T_h,2] or [B,T_h,2]; "
            f"received {positions_raw.shape}."
        )
    if velocities_raw.shape != positions_raw.shape:
        raise ValueError(
            f"history_velocity must have shape {positions_raw.shape}; "
            f"received {velocities_raw.shape}."
        )
    if mask.dtype.kind != "b" or mask.shape != positions_raw.shape[:-1]:
        raise ValueError(
            "history_mask must be boolean and match history_position without its "
            f"coordinate dimension; received dtype {mask.dtype} and shape {mask.shape}."
        )
    if history_time_raw.shape != positions_raw.shape[:-1]:
        raise ValueError(
            "history_timestamps_ns must match history_position's batch/time dimensions; "
            f"received {history_time_raw.shape} and {positions_raw.shape[:-1]}."
        )

    unbatched = positions_raw.ndim == 2
    if unbatched:
        if future_time_raw.ndim != 1:
            raise ValueError(
                "future_timestamps_ns must have shape [T_f] for an unbatched input; "
                f"received {future_time_raw.shape}."
            )
        positions_raw = positions_raw[None, :, :]
        velocities_raw = velocities_raw[None, :, :]
        mask = mask[None, :]
        history_time_raw = history_time_raw[None, :]
        future_time_raw = future_time_raw[None, :]
    elif future_time_raw.ndim != 2 or future_time_raw.shape[0] != positions_raw.shape[0]:
        raise ValueError(
            "future_timestamps_ns must have shape [B,T_f] matching the batch; "
            f"received {future_time_raw.shape} for B={positions_raw.shape[0]}."
        )

    batch_size, history_steps, _ = positions_raw.shape
    future_steps = future_time_raw.shape[1]
    if batch_size <= 0 or history_steps <= 0 or future_steps <= 0:
        raise ValueError(
            "CV batch, history, and future dimensions must all be positive; "
            f"received B={batch_size}, T_h={history_steps}, T_f={future_steps}."
        )

    positions = _as_float64_array(positions_raw, name="history_position")
    velocities = _as_float64_array(velocities_raw, name="history_velocity")
    history_time = _as_float64_array(
        history_time_raw, name="history_timestamps_ns"
    )
    future_time = _as_float64_array(future_time_raw, name="future_timestamps_ns")

    invalid_samples = np.flatnonzero(~mask.any(axis=1)).tolist()
    if invalid_samples:
        raise ValueError(
            "Every sample needs at least one valid historical state; "
            f"samples without valid history are at indices {invalid_samples}."
        )

    bad_history_time = np.flatnonzero(
        np.any(np.diff(history_time, axis=1) <= 0.0, axis=1)
    ).tolist()
    if bad_history_time:
        raise ValueError(
            "history_timestamps_ns must be strictly increasing; "
            f"violations are at sample indices {bad_history_time}."
        )
    bad_future_time = np.flatnonzero(
        np.any(np.diff(future_time, axis=1) <= 0.0, axis=1)
    ).tolist()
    if bad_future_time:
        raise ValueError(
            "future_timestamps_ns must be strictly increasing; "
            f"violations are at sample indices {bad_future_time}."
        )
    bad_boundary = np.flatnonzero(future_time[:, 0] <= history_time[:, -1]).tolist()
    if bad_boundary:
        raise ValueError(
            "Every future timestamp must be after the history window; "
            f"violations are at sample indices {bad_boundary}."
        )

    return positions, velocities, mask, history_time, future_time, unbatched


def predict_constant_velocity(
    history_position: Any,
    history_velocity: Any,
    history_mask: Any,
    history_timestamps_ns: Any,
    future_timestamps_ns: Any,
) -> np.ndarray:
    """Extrapolate a causal CV trajectory at requested nanosecond timestamps.

    Inputs are one trajectory ``[T_h,2]``/``[T_h]``/``[T_f]`` or a batch
    ``[B,T_h,2]``/``[B,T_h]``/``[B,T_f]``. Positions use metres, velocities
    metres/second, masks boolean validity, and timestamps nanoseconds. The
    float64 output is ``[T_f,2]`` or ``[B,T_f,2]`` in the input coordinate
    frame. Only the final valid historical position, velocity, and time are
    used; future state labels are not inputs.
    """

    positions, velocities, mask, history_time, future_time, unbatched = (
        _normalize_cv_inputs(
            history_position,
            history_velocity,
            history_mask,
            history_timestamps_ns,
            future_timestamps_ns,
        )
    )
    batch_size, history_steps, _ = positions.shape
    last_valid = history_steps - 1 - np.argmax(mask[:, ::-1], axis=1)
    batch_indices = np.arange(batch_size)
    reference_position = positions[batch_indices, last_valid]
    reference_velocity = velocities[batch_indices, last_valid]
    reference_time = history_time[batch_indices, last_valid]
    delta_seconds = (
        future_time - reference_time[:, None]
    ) / NANOSECONDS_PER_SECOND
    prediction = (
        reference_position[:, None, :]
        + reference_velocity[:, None, :] * delta_seconds[:, :, None]
    )
    if not np.isfinite(prediction).all():
        raise ValueError("Constant-velocity extrapolation produced non-finite positions.")
    return prediction[0] if unbatched else prediction


def _as_timestep_indices(value: Any, *, name: str) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional array of integer indices.")
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one index.")
    return indices.astype(np.int64, copy=False)


def predict_constant_velocity_sample(sample: TrajectorySample) -> np.ndarray:
    """Predict ``[T_f,2]`` metres for a sample without reading future labels.

    The function reads historical position, velocity, mask, and the timestamp
    entries addressed by history/future timestep indices. The float64 result
    remains in ``sample.coordinate_frame``; future positions, velocities,
    headings, observed flags, and masks do not affect it.
    """

    if not isinstance(sample, TrajectorySample):
        raise TypeError("sample must be a TrajectorySample instance.")
    timestamps = np.asarray(sample.timestamps_ns)
    if timestamps.ndim != 1:
        raise ValueError(
            f"timestamps_ns must be one-dimensional; received {timestamps.shape}."
        )
    _as_float64_array(timestamps, name="timestamps_ns")
    history_indices = _as_timestep_indices(
        sample.history_timesteps, name="history_timesteps"
    )
    future_indices = _as_timestep_indices(
        sample.future_timesteps, name="future_timesteps"
    )
    all_indices = np.concatenate((history_indices, future_indices))
    if np.any(all_indices < 0) or np.any(all_indices >= timestamps.size):
        raise ValueError(
            "history_timesteps and future_timesteps contain indices outside timestamps_ns."
        )
    return predict_constant_velocity(
        sample.history_position,
        sample.history_velocity,
        sample.history_mask,
        timestamps[history_indices],
        timestamps[future_indices],
    )
