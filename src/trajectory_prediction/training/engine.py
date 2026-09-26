"""Shared train/validation engine interface."""

from __future__ import annotations


class TrainingEngine:
    """Own model optimization, validation, logging, and resumable checkpoints."""

    def fit(self, *args: object, **kwargs: object) -> object:
        """Train from scratch or resume without overwriting an existing run."""

        raise NotImplementedError("Training is implemented with the multimodal baseline.")
