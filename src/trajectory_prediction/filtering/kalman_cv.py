"""Online two-dimensional constant-velocity Kalman filter interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KalmanCVConfig:
    """Numerical parameters for a causal CV Kalman filter."""

    process_acceleration_std_mps2: float
    observation_position_std_m: float
    covariance_update: str = "joseph"


class KalmanCV:
    """Estimate [x, y, vx, vy] using actual time deltas and masked updates."""

    def __init__(self, config: KalmanCVConfig) -> None:
        self.config = config

    def filter(self, *args: object, **kwargs: object) -> object:
        """Run forward-only filtering; missing observations perform prediction only."""

        raise NotImplementedError("Kalman filtering is implemented in the state-estimation milestone.")
