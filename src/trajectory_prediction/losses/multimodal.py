"""Best-of-K trajectory and mode-classification loss interface."""

from __future__ import annotations


def multimodal_loss(*args: object, **kwargs: object) -> object:
    """Select the best valid mode and combine Smooth L1 with mode cross-entropy."""

    raise NotImplementedError("Multimodal loss is implemented with the GRU baseline.")
