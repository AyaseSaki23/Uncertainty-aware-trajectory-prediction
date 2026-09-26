"""BEV trajectory, mode probability, and covariance ellipse interfaces."""

from __future__ import annotations


def plot_prediction_bev(*args: object, **kwargs: object) -> object:
    """Render observations, KF states, covariance, K modes, and ground truth."""

    raise NotImplementedError("BEV rendering is implemented with the first AV2 sample reader.")
