"""Confidence features derived from Kalman posterior and innovation statistics."""

from __future__ import annotations


def extract_confidence_features(*args: object, **kwargs: object) -> object:
    """Build bounded log-covariance, NIS, mask, and missing-duration features."""

    raise NotImplementedError("Confidence features are implemented after KF validation.")
