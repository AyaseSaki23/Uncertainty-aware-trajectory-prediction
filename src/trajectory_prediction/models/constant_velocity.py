"""Deterministic constant-velocity forecasting baseline."""

from __future__ import annotations


def predict_constant_velocity(*args: object, **kwargs: object) -> object:
    """Extrapolate from the last valid causal state using requested timestamps."""

    raise NotImplementedError("The CV baseline is implemented with metric validation.")
