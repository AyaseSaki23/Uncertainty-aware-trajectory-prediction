"""Causal global and target-local coordinate transforms for trajectory samples."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from trajectory_prediction.contracts import TrajectorySample


def _as_float_array(value: Any, *, name: str, last_dimension_two: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values; received dtype {array.dtype}.")
    if last_dimension_two and (array.ndim == 0 or array.shape[-1] != 2):
        raise ValueError(f"{name} must have last dimension 2; received shape {array.shape}.")
    if array.dtype.kind == "f" and array.dtype.itemsize > 4:
        dtype = np.dtype(np.float64)
    elif array.dtype.kind == "f":
        dtype = np.dtype(np.float32)
    else:
        dtype = np.dtype(np.float64)
    converted = array.astype(dtype, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values.")
    return converted


def _as_origin(origin_xy: Any, dtype: np.dtype[Any]) -> np.ndarray:
    origin = np.asarray(origin_xy)
    if origin.shape != (2,):
        raise ValueError(f"origin_xy must have shape (2,); received shape {origin.shape}.")
    if origin.dtype.kind not in "iuf":
        raise TypeError(
            f"origin_xy must contain real numeric values; received dtype {origin.dtype}."
        )
    converted = origin.astype(dtype, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError("origin_xy must contain only finite values.")
    return converted


def _as_reference_heading(reference_heading_rad: Any, dtype: np.dtype[Any]) -> np.floating[Any]:
    heading = np.asarray(reference_heading_rad)
    if heading.shape != () or heading.dtype.kind not in "iuf":
        raise ValueError("reference_heading_rad must be one finite real scalar in radians.")
    converted = heading.astype(dtype, copy=False)[()]
    if not np.isfinite(converted):
        raise ValueError("reference_heading_rad must be finite.")
    return converted


def _rotate_vectors(
    vectors: Any, reference_heading_rad: Any, *, global_to_target: bool
) -> np.ndarray:
    array = _as_float_array(vectors, name="vectors", last_dimension_two=True)
    heading = _as_reference_heading(reference_heading_rad, array.dtype)
    cosine = np.asarray(np.cos(heading), dtype=array.dtype)
    sine = np.asarray(np.sin(heading), dtype=array.dtype)
    x_values = array[..., 0]
    y_values = array[..., 1]
    if global_to_target:
        rotated_x = cosine * x_values + sine * y_values
        rotated_y = -sine * x_values + cosine * y_values
    else:
        rotated_x = cosine * x_values - sine * y_values
        rotated_y = sine * x_values + cosine * y_values
    return np.stack((rotated_x, rotated_y), axis=-1)


def wrap_angle(angle_rad: Any) -> np.ndarray | np.floating[Any]:
    """Wrap scalar or array angles in radians to the half-open range ``[-pi, pi)``."""

    angles = _as_float_array(angle_rad, name="angle_rad")
    pi = np.asarray(np.pi, dtype=angles.dtype)
    two_pi = np.asarray(2.0 * np.pi, dtype=angles.dtype)
    wrapped = (angles + pi) % two_pi - pi
    return wrapped[()] if wrapped.ndim == 0 else wrapped


def global_to_local(
    points: Any, origin_xy: Any, reference_heading_rad: Any
) -> np.ndarray:
    """Map global positions ``[..., 2]`` in metres into the target-local frame."""

    positions = _as_float_array(points, name="points", last_dimension_two=True)
    origin = _as_origin(origin_xy, positions.dtype)
    return _rotate_vectors(
        positions - origin, reference_heading_rad, global_to_target=True
    )


def local_to_global(
    points: Any, origin_xy: Any, reference_heading_rad: Any
) -> np.ndarray:
    """Map target-local positions ``[..., 2]`` in metres back to the global frame."""

    positions = _as_float_array(points, name="points", last_dimension_two=True)
    origin = _as_origin(origin_xy, positions.dtype)
    return (
        _rotate_vectors(positions, reference_heading_rad, global_to_target=False) + origin
    )


def global_vector_to_local(vectors: Any, reference_heading_rad: Any) -> np.ndarray:
    """Rotate global vectors ``[..., 2]`` into the target frame without translation."""

    return _rotate_vectors(vectors, reference_heading_rad, global_to_target=True)


def local_vector_to_global(vectors: Any, reference_heading_rad: Any) -> np.ndarray:
    """Rotate target-local vectors ``[..., 2]`` back into the global frame."""

    return _rotate_vectors(vectors, reference_heading_rad, global_to_target=False)


def global_heading_to_local(headings_rad: Any, reference_heading_rad: Any) -> Any:
    """Express global headings in radians relative to the target reference heading."""

    headings = _as_float_array(headings_rad, name="headings_rad")
    reference = _as_reference_heading(reference_heading_rad, headings.dtype)
    return wrap_angle(headings - reference)


def local_heading_to_global(headings_rad: Any, reference_heading_rad: Any) -> Any:
    """Restore target-local headings in radians to the global frame."""

    headings = _as_float_array(headings_rad, name="headings_rad")
    reference = _as_reference_heading(reference_heading_rad, headings.dtype)
    return wrap_angle(headings + reference)


def _validate_sample_window(
    *,
    position: Any,
    velocity: Any,
    heading: Any,
    mask: Any,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = _as_float_array(
        position, name=f"{prefix}_position", last_dimension_two=True
    )
    velocities = _as_float_array(
        velocity, name=f"{prefix}_velocity", last_dimension_two=True
    )
    headings = _as_float_array(heading, name=f"{prefix}_heading")
    masks = np.asarray(mask)
    expected_length = positions.shape[0] if positions.ndim == 2 else None
    if positions.ndim != 2:
        raise ValueError(
            f"{prefix}_position must have shape [T, 2]; received {positions.shape}."
        )
    if velocities.shape != positions.shape:
        raise ValueError(
            f"{prefix}_velocity must have shape {positions.shape}; received {velocities.shape}."
        )
    if headings.shape != (expected_length,):
        raise ValueError(
            f"{prefix}_heading must have shape ({expected_length},); received {headings.shape}."
        )
    if masks.dtype.kind != "b" or masks.shape != (expected_length,):
        raise ValueError(
            f"{prefix}_mask must be boolean with shape ({expected_length},); "
            f"received dtype {masks.dtype} and shape {masks.shape}."
        )
    return positions, velocities, headings, masks


def _masked_window_to_local(
    position: Any,
    velocity: Any,
    heading: Any,
    mask: Any,
    *,
    prefix: str,
    origin_xy: Any,
    reference_heading_rad: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, velocities, headings, masks = _validate_sample_window(
        position=position,
        velocity=velocity,
        heading=heading,
        mask=mask,
        prefix=prefix,
    )
    local_position = np.zeros_like(positions)
    local_velocity = np.zeros_like(velocities)
    local_heading = np.zeros_like(headings)
    if masks.any():
        local_position[masks] = global_to_local(
            positions[masks], origin_xy, reference_heading_rad
        )
        local_velocity[masks] = global_vector_to_local(
            velocities[masks], reference_heading_rad
        )
        local_heading[masks] = global_heading_to_local(
            headings[masks], reference_heading_rad
        )
    return local_position, local_velocity, local_heading


def _masked_window_to_global(
    position: Any,
    velocity: Any,
    heading: Any,
    mask: Any,
    *,
    prefix: str,
    origin_xy: Any,
    reference_heading_rad: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, velocities, headings, masks = _validate_sample_window(
        position=position,
        velocity=velocity,
        heading=heading,
        mask=mask,
        prefix=prefix,
    )
    global_position = np.zeros_like(positions)
    global_velocity = np.zeros_like(velocities)
    global_heading = np.zeros_like(headings)
    if masks.any():
        global_position[masks] = local_to_global(
            positions[masks], origin_xy, reference_heading_rad
        )
        global_velocity[masks] = local_vector_to_global(
            velocities[masks], reference_heading_rad
        )
        global_heading[masks] = local_heading_to_global(
            headings[masks], reference_heading_rad
        )
    return global_position, global_velocity, global_heading


def _copy_sample_metadata(sample: TrajectorySample) -> dict[str, np.ndarray]:
    return {
        "history_timesteps": np.asarray(sample.history_timesteps).copy(),
        "future_timesteps": np.asarray(sample.future_timesteps).copy(),
        "timestamps_ns": np.asarray(sample.timestamps_ns).copy(),
        "history_observed": np.asarray(sample.history_observed).copy(),
        "history_mask": np.asarray(sample.history_mask).copy(),
        "future_observed": np.asarray(sample.future_observed).copy(),
        "future_mask": np.asarray(sample.future_mask).copy(),
        "origin_xy": np.asarray(sample.origin_xy).copy(),
    }


def trajectory_sample_to_local(sample: TrajectorySample) -> TrajectorySample:
    """Return a new local-frame sample using only its stored historical reference state."""

    if sample.coordinate_frame != "global":
        raise ValueError(
            "trajectory_sample_to_local requires coordinate_frame='global'; "
            f"received {sample.coordinate_frame!r}."
        )
    history_mask = np.asarray(sample.history_mask)
    if history_mask.dtype.kind != "b" or history_mask.ndim != 1:
        raise ValueError("history_mask must be a one-dimensional boolean array.")
    if not history_mask.any():
        raise ValueError(
            "Cannot define a target frame because the sample has no valid historical state."
        )

    history_position = _as_float_array(
        sample.history_position, name="history_position", last_dimension_two=True
    )
    history_heading = _as_float_array(sample.history_heading, name="history_heading")
    last_valid = int(np.flatnonzero(history_mask)[-1])
    origin = _as_origin(sample.origin_xy, history_position.dtype)
    reference = _as_reference_heading(sample.reference_heading_rad, history_heading.dtype)
    if not np.allclose(history_position[last_valid], origin, rtol=1e-6, atol=1e-6):
        raise ValueError("origin_xy does not match the last valid historical position.")
    heading_error = float(wrap_angle(history_heading[last_valid] - reference))
    if abs(heading_error) > 1e-6:
        raise ValueError(
            "reference_heading_rad does not match the last valid historical heading."
        )

    local_history = _masked_window_to_local(
        sample.history_position,
        sample.history_velocity,
        sample.history_heading,
        sample.history_mask,
        prefix="history",
        origin_xy=origin,
        reference_heading_rad=reference,
    )
    local_future = _masked_window_to_local(
        sample.future_position,
        sample.future_velocity,
        sample.future_heading,
        sample.future_mask,
        prefix="future",
        origin_xy=origin,
        reference_heading_rad=reference,
    )
    return replace(
        sample,
        **_copy_sample_metadata(sample),
        history_position=local_history[0],
        history_velocity=local_history[1],
        history_heading=local_history[2],
        future_position=local_future[0],
        future_velocity=local_future[1],
        future_heading=local_future[2],
        coordinate_frame="local",
    )


def trajectory_sample_to_global(sample: TrajectorySample) -> TrajectorySample:
    """Return a new global-frame sample using reference metadata retained during localization."""

    if sample.coordinate_frame != "local":
        raise ValueError(
            "trajectory_sample_to_global requires coordinate_frame='local'; "
            f"received {sample.coordinate_frame!r}."
        )
    history_mask = np.asarray(sample.history_mask)
    if history_mask.dtype.kind != "b" or history_mask.ndim != 1 or not history_mask.any():
        raise ValueError("A local sample must contain at least one valid historical state.")
    history_position = _as_float_array(
        sample.history_position, name="history_position", last_dimension_two=True
    )
    history_heading = _as_float_array(sample.history_heading, name="history_heading")
    origin = _as_origin(sample.origin_xy, history_position.dtype)
    reference = _as_reference_heading(sample.reference_heading_rad, history_heading.dtype)
    global_history = _masked_window_to_global(
        sample.history_position,
        sample.history_velocity,
        sample.history_heading,
        sample.history_mask,
        prefix="history",
        origin_xy=origin,
        reference_heading_rad=reference,
    )
    global_future = _masked_window_to_global(
        sample.future_position,
        sample.future_velocity,
        sample.future_heading,
        sample.future_mask,
        prefix="future",
        origin_xy=origin,
        reference_heading_rad=reference,
    )
    return replace(
        sample,
        **_copy_sample_metadata(sample),
        history_position=global_history[0],
        history_velocity=global_history[1],
        history_heading=global_history[2],
        future_position=global_future[0],
        future_velocity=global_future[1],
        future_heading=global_future[2],
        coordinate_frame="global",
    )
