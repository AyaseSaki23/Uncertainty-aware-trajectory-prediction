"""Bird's-eye-view visualizations for trajectory samples and predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from trajectory_prediction.contracts import TrajectorySample

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_focal_sample_bev(
    sample: TrajectorySample, output_path: str | Path | None = None
) -> object:
    """Plot one focal track in global AV2 coordinates and optionally save a PNG."""

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    history = np.asarray(sample.history_position)[np.asarray(sample.history_mask, dtype=bool)]
    future = np.asarray(sample.future_position)[np.asarray(sample.future_mask, dtype=bool)]
    if len(history):
        axis.plot(history[:, 0], history[:, 1], "o-", color="#2563eb", label="history", ms=3)
        axis.scatter(
            history[-1, 0],
            history[-1, 1],
            s=90,
            facecolors="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="history/future boundary",
        )
    if len(future):
        axis.plot(future[:, 0], future[:, 1], "o-", color="#f97316", label="ground truth", ms=3)

    missing_history = np.asarray(sample.history_timesteps)[
        ~np.asarray(sample.history_mask, dtype=bool)
    ]
    missing_future = np.asarray(sample.future_timesteps)[
        ~np.asarray(sample.future_mask, dtype=bool)
    ]
    missing_text = (
        f"missing history: {missing_history.tolist()}\n"
        f"missing future: {missing_future.tolist()}\n"
        "missing states are masked; their positions are not interpolated"
    )
    axis.text(
        0.01,
        0.01,
        missing_text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#991b1b",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("global x (m)")
    axis.set_ylabel("global y (m)")
    axis.set_title(f"AV2 {sample.split} | scenario {sample.scenario_id}")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        return destination
    return figure


def plot_prediction_bev(
    sample: TrajectorySample, output_path: str | Path | None = None
) -> object:
    """Compatibility entry point; Week 1 renders the focal sample only."""

    return plot_focal_sample_bev(sample, output_path)
