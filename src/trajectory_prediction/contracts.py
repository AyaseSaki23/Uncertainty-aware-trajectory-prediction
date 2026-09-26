"""Shared sample and prediction contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectorySample:
    """One causal AV2 forecasting sample in global coordinates.

    Position and velocity arrays use metres and metres/second. Heading uses
    radians. Missing states contain zero-valued placeholders and must only be
    consumed through their corresponding boolean masks.
    """

    scenario_id: str
    split: str
    focal_track_id: str
    city: str
    history_timesteps: Any
    future_timesteps: Any
    timestamps_ns: Any
    history_position: Any
    history_velocity: Any
    history_heading: Any
    history_observed: Any
    history_mask: Any
    future_position: Any
    future_velocity: Any
    future_heading: Any
    future_observed: Any
    future_mask: Any
    origin_xy: Any
    reference_heading_rad: float


@dataclass(frozen=True)
class MultimodalPrediction:
    """K future trajectories and their normalized mode probabilities."""

    trajectories: Any
    mode_probabilities: Any
