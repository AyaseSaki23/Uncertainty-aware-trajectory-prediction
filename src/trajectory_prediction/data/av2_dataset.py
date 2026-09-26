"""Argoverse 2 Motion Forecasting dataset adapter.

The implementation will preserve official splits, actual timestamps, masks, and
scenario identifiers. It intentionally does not download any AV2 data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from trajectory_prediction.contracts import TrajectorySample


class AV2MotionDataset:
    """Lazy AV2 scenario reader exposing the unified trajectory sample contract."""

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; received {split!r}")

    def __iter__(self) -> Iterator[TrajectorySample]:
        raise NotImplementedError("AV2 parsing is scheduled for the data-pipeline milestone.")
