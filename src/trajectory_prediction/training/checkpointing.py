"""Best/last checkpoint and experiment metadata interfaces."""

from __future__ import annotations


def save_checkpoint(*args: object, **kwargs: object) -> None:
    """Persist model, optimizer, scheduler, scaler, epoch, and RNG state."""

    raise NotImplementedError("Checkpoint persistence is implemented with the training engine.")
