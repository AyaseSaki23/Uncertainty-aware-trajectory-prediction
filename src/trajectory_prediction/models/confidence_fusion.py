"""Dual-branch confidence gating interface."""

from __future__ import annotations


class ConfidenceFusion:
    """Fuse motion and confidence embeddings without changing the shared backbone."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Confidence fusion is implemented in the ablation milestone.")
