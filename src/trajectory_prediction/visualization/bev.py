"""Bird's-eye-view visualizations for trajectory samples and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.geometry.coordinates import trajectory_sample_to_local

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _validated_path(points: Any, mask: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(points)
    valid = np.asarray(mask)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"{name} points must have shape [T,2]; received {positions.shape}.")
    if valid.dtype.kind != "b" or valid.shape != (positions.shape[0],):
        raise ValueError(
            f"{name} mask must be boolean with shape ({positions.shape[0]},); "
            f"received dtype {valid.dtype} and shape {valid.shape}."
        )
    positions = positions.astype(np.float64, copy=False)
    if not np.isfinite(positions).all():
        raise ValueError(f"{name} points must contain only finite values.")
    return positions, valid


def _plot_masked_path(
    axis: Any,
    points: Any,
    mask: Any,
    *,
    name: str,
    color: str,
    label: str,
) -> None:
    positions, valid = _validated_path(points, mask, name=name)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
    runs = np.split(valid_indices, boundaries)
    for run_index, indices in enumerate(runs):
        (line,) = axis.plot(
            positions[indices, 0],
            positions[indices, 1],
            "o-",
            color=color,
            label=label if run_index == 0 else "_nolegend_",
            ms=3,
            linewidth=1.7,
        )
        line.set_gid(name)


def _mark_history_boundary(axis: Any, points: Any, mask: Any) -> None:
    positions, valid = _validated_path(points, mask, name="history")
    if not valid.any():
        return
    last = int(np.flatnonzero(valid)[-1])
    axis.scatter(
        positions[last, 0],
        positions[last, 1],
        s=90,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label="history/future boundary",
    )


def _configure_axis(axis: Any, *, frame: str) -> None:
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel(f"{frame} x (m)")
    axis.set_ylabel(f"{frame} y (m)")
    axis.set_title(f"{frame.capitalize()} coordinates")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")


def _plot_prediction_panel(
    axis: Any,
    *,
    history_position: Any,
    history_mask: Any,
    future_position: Any,
    future_mask: Any,
    prediction: Any,
    frame: str,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float64)
    future = np.asarray(future_position)
    if predicted.shape != future.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError(
            f"{frame} prediction must match future_position shape {future.shape}; "
            f"received {predicted.shape}."
        )
    if not np.isfinite(predicted).all():
        raise ValueError(f"{frame} prediction must contain only finite values.")
    _plot_masked_path(
        axis,
        history_position,
        history_mask,
        name="history",
        color="#2563eb",
        label="history",
    )
    _plot_masked_path(
        axis,
        future_position,
        future_mask,
        name="ground_truth",
        color="#f97316",
        label="ground truth",
    )
    (prediction_line,) = axis.plot(
        predicted[:, 0],
        predicted[:, 1],
        "o--",
        color="#16a34a",
        label="CV prediction",
        ms=3,
        linewidth=1.7,
    )
    prediction_line.set_gid("cv_prediction")
    _mark_history_boundary(axis, history_position, history_mask)
    _configure_axis(axis, frame=frame)


def plot_focal_sample_bev(
    sample: TrajectorySample, output_path: str | Path | None = None
) -> object:
    """Plot one focal track in global AV2 coordinates and optionally save a PNG."""

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    _plot_masked_path(
        axis,
        sample.history_position,
        sample.history_mask,
        name="history",
        color="#2563eb",
        label="history",
    )
    _plot_masked_path(
        axis,
        sample.future_position,
        sample.future_mask,
        name="ground_truth",
        color="#f97316",
        label="ground truth",
    )
    _mark_history_boundary(axis, sample.history_position, sample.history_mask)

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


def plot_prediction_comparison(
    sample: TrajectorySample,
    local_prediction_m: Any,
    global_prediction_m: Any,
    metrics: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> object:
    """Plot global/local history, ground truth, and CV prediction in metres.

    ``sample`` must be global, predictions must be ``[T_f,2]``, and metrics
    must contain metre-valued ``ADE`` and ``FDE``. Masked observations are
    drawn as separate contiguous runs so gaps are never presented as data.
    """

    if sample.coordinate_frame != "global":
        raise ValueError("plot_prediction_comparison requires a global TrajectorySample.")
    try:
        ade = float(metrics["ADE"])
        fde = float(metrics["FDE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metrics must contain finite numeric ADE and FDE values.") from exc
    if not np.isfinite([ade, fde]).all():
        raise ValueError("metrics ADE and FDE must be finite.")
    local_sample = trajectory_sample_to_local(sample)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)
    _plot_prediction_panel(
        axes[0],
        history_position=sample.history_position,
        history_mask=sample.history_mask,
        future_position=sample.future_position,
        future_mask=sample.future_mask,
        prediction=global_prediction_m,
        frame="global",
    )
    _plot_prediction_panel(
        axes[1],
        history_position=local_sample.history_position,
        history_mask=local_sample.history_mask,
        future_position=local_sample.future_position,
        future_mask=local_sample.future_mask,
        prediction=local_prediction_m,
        frame="local",
    )
    figure.suptitle(
        f"AV2 {sample.split} | scenario {sample.scenario_id} | "
        f"CV ADE={ade:.3f} m, FDE={fde:.3f} m"
    )
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(f"Visualization already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=170)
        plt.close(figure)
        return destination
    return figure


def plot_prediction_bev(
    sample: TrajectorySample, output_path: str | Path | None = None
) -> object:
    """Compatibility entry point that renders the focal sample only."""

    return plot_focal_sample_bev(sample, output_path)
