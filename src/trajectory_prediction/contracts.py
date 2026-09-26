"""Shared sample and prediction contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectorySample:
    """One causal forecasting sample; array shapes are documented in data_contract.md."""

    scenario_id: str
    timestamps_ns: Any
    history_position: Any
    history_velocity: Any
    history_mask: Any
    future_position: Any
    future_mask: Any
    origin_xy: Any
    heading_rad: float


@dataclass(frozen=True)
class MultimodalPrediction:
    """K future trajectories and their normalized mode probabilities."""

    trajectories: Any
    mode_probabilities: Any
