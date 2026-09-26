"""Robustness degradation, confidence correlation, and AUROC interfaces."""

from __future__ import annotations


def compute_robustness_metrics(*args: object, **kwargs: object) -> dict[str, float]:
    """Compare clean and degraded evaluations using aligned scenario identifiers."""

    raise NotImplementedError("Robustness metrics are implemented before formal ablations.")
