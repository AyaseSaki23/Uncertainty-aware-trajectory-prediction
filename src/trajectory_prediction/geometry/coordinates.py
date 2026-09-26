"""Global and target-local coordinate transform interfaces."""

from __future__ import annotations


def global_to_local(*args: object, **kwargs: object) -> object:
    """Transform points to a frame centered at the last valid target position."""

    raise NotImplementedError("Coordinate transforms are implemented in the baseline milestone.")


def local_to_global(*args: object, **kwargs: object) -> object:
    """Invert :func:`global_to_local` for positions or predicted trajectories."""

    raise NotImplementedError("Coordinate transforms are implemented in the baseline milestone.")
