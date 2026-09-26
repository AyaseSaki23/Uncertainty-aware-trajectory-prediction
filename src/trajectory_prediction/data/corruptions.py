"""Causal history-only observation degradation interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorruptionConfig:
    """Parameters for reproducible position noise and missing observations."""

    position_std_m: float = 0.0
    random_missing_rate: float = 0.0
    contiguous_missing_steps: int = 0
    seed: int = 2026
    keep_last_observation: bool = True


def corrupt_history(*args: object, **kwargs: object) -> object:
    """Apply configured degradation only to historical observations."""

    raise NotImplementedError("Observation degradation is implemented with the KF milestone.")
