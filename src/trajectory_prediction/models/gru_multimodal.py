"""Lightweight GRU multimodal predictor interface."""

from __future__ import annotations


class GRUMultimodalPredictor:
    """Encode causal motion history and decode K trajectories plus mode logits."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("The learned baseline is implemented after CV/KF validation.")
