"""ADE, FDE, minADE, minFDE, Miss Rate, and probability metric interfaces."""

from __future__ import annotations


def compute_forecasting_metrics(*args: object, **kwargs: object) -> dict[str, float]:
    """Compute mask-aware metrics with explicit mode handling."""

    raise NotImplementedError("Metrics are implemented and hand-checked in milestone two.")
