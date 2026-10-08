"""Physical and learned trajectory prediction models."""

from trajectory_prediction.models.constant_velocity import (
    predict_constant_velocity,
    predict_constant_velocity_sample,
)

__all__ = ["predict_constant_velocity", "predict_constant_velocity_sample"]
