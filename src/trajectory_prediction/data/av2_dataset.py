"""Strict, dependency-light Argoverse 2 Motion Forecasting data pipeline.

The official AV2 serializer stores scenario metadata on every Parquet row. This
module intentionally reads that representation with pandas/pyarrow so Week 1
does not require the full AV2 package or any sensor-data dependencies.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from trajectory_prediction.contracts import TrajectorySample


VALID_SPLITS = frozenset({"train", "val", "test"})
REQUIRED_SCENARIO_COLUMNS = frozenset(
    {
        "observed",
        "track_id",
        "object_type",
        "object_category",
        "timestep",
        "position_x",
        "position_y",
        "heading",
        "velocity_x",
        "velocity_y",
        "scenario_id",
        "start_timestamp",
        "end_timestamp",
        "num_timestamps",
        "focal_track_id",
        "city",
    }
)
STATE_COLUMNS = ("position_x", "position_y", "velocity_x", "velocity_y", "heading")


class ScenarioValidationError(ValueError):
    """Raised when an AV2 scenario violates the expected data contract."""


@dataclass(frozen=True)
class AV2Scenario:
    """Validated rows and metadata for one AV2 Motion Forecasting scenario."""

    scenario_id: str
    focal_track_id: str
    city: str
    timestamps_ns: np.ndarray
    tracks: pd.DataFrame
    source_path: Path | None = None


def _one_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ScenarioValidationError(
            f"Column {column!r} must contain exactly one scenario-level value; found {values!r}."
        )
    return values[0]


def validate_scenario_dataframe(
    frame: pd.DataFrame, *, source_path: str | Path | None = None
) -> AV2Scenario:
    """Validate an official AV2 scenario table and reconstruct real timestamps."""

    source = Path(source_path) if source_path is not None else None
    missing_columns = sorted(REQUIRED_SCENARIO_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ScenarioValidationError(
            f"AV2 scenario {source or '<memory>'} is missing columns: {', '.join(missing_columns)}"
        )
    if frame.empty:
        raise ScenarioValidationError(f"AV2 scenario {source or '<memory>'} contains no states.")

    scenario_id = str(_one_value(frame, "scenario_id"))
    focal_track_id = str(_one_value(frame, "focal_track_id"))
    city = str(_one_value(frame, "city"))
    start_timestamp = int(_one_value(frame, "start_timestamp"))
    end_timestamp = int(_one_value(frame, "end_timestamp"))
    num_timestamps = int(_one_value(frame, "num_timestamps"))
    if num_timestamps < 2:
        raise ScenarioValidationError(
            f"Scenario {scenario_id} must have at least two timestamps; found {num_timestamps}."
        )
    if end_timestamp <= start_timestamp:
        raise ScenarioValidationError(
            f"Scenario {scenario_id} has non-increasing timestamp bounds: "
            f"{start_timestamp} -> {end_timestamp}."
        )

    numeric_timestep = pd.to_numeric(frame["timestep"], errors="coerce")
    integer_timesteps = np.equal(numeric_timestep, np.floor(numeric_timestep)).all()
    if numeric_timestep.isna().any() or not integer_timesteps:
        raise ScenarioValidationError(f"Scenario {scenario_id} contains non-integer timesteps.")
    timesteps = numeric_timestep.astype(np.int64)
    if ((timesteps < 0) | (timesteps >= num_timestamps)).any():
        bad = sorted(timesteps[(timesteps < 0) | (timesteps >= num_timestamps)].unique().tolist())
        raise ScenarioValidationError(
            f"Scenario {scenario_id} contains timesteps outside [0, {num_timestamps - 1}]: {bad}."
        )

    duplicate_rows = frame.assign(timestep=timesteps).duplicated(
        subset=["track_id", "timestep"], keep=False
    )
    if duplicate_rows.any():
        duplicate = frame.loc[duplicate_rows, ["track_id", "timestep"]].iloc[0]
        raise ScenarioValidationError(
            f"Scenario {scenario_id} has duplicate timestep {int(duplicate['timestep'])} "
            f"for track {duplicate['track_id']!r}."
        )

    numeric_states = frame.loc[:, STATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric_states.to_numpy(dtype=np.float64, copy=False))
    if not finite.all():
        row_index, column_index = np.argwhere(~finite)[0]
        raise ScenarioValidationError(
            f"Scenario {scenario_id} contains a NaN or infinite value in "
            f"{STATE_COLUMNS[int(column_index)]!r} at table row {frame.index[int(row_index)]!r}."
        )

    if frame["observed"].isna().any():
        raise ScenarioValidationError(f"Scenario {scenario_id} contains null observed flags.")
    if not frame["observed"].isin([True, False, 0, 1]).all():
        raise ScenarioValidationError(f"Scenario {scenario_id} contains invalid observed flags.")

    normalized = frame.copy()
    normalized["timestep"] = timesteps
    normalized.loc[:, STATE_COLUMNS] = numeric_states
    normalized["observed"] = normalized["observed"].astype(bool)
    normalized = normalized.sort_values(["track_id", "timestep"], kind="stable").reset_index(
        drop=True
    )

    focal_rows = normalized[normalized["track_id"].astype(str) == focal_track_id]
    if focal_rows.empty:
        raise ScenarioValidationError(
            f"Scenario {scenario_id} does not contain focal_track_id {focal_track_id!r}."
        )

    # AV2 stores timestamp bounds and a count. Recreate the official integer
    # nanosecond timeline instead of assuming a fixed 0.1-second interval.
    timestamps_ns = np.linspace(
        start_timestamp, end_timestamp, num=num_timestamps, dtype=np.int64
    )
    if not np.all(np.diff(timestamps_ns) > 0):
        raise ScenarioValidationError(
            f"Scenario {scenario_id} reconstructs to non-monotonic or repeated timestamps."
        )
    return AV2Scenario(
        scenario_id=scenario_id,
        focal_track_id=focal_track_id,
        city=city,
        timestamps_ns=timestamps_ns,
        tracks=normalized,
        source_path=source,
    )


def read_av2_scenario(path: str | Path) -> AV2Scenario:
    """Read and validate one official ``scenario_*.parquet`` file."""

    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"AV2 scenario parquet does not exist: {scenario_path}")
    try:
        frame = pd.read_parquet(scenario_path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading AV2 Parquet requires pyarrow. Install it in the active environment."
        ) from exc
    return validate_scenario_dataframe(frame, source_path=scenario_path)


def _empty_state_arrays(length: int) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((length, 2), dtype=np.float32),
        np.zeros((length, 2), dtype=np.float32),
        np.zeros(length, dtype=np.float32),
        np.zeros(length, dtype=bool),
        np.zeros(length, dtype=bool),
    )


def _align_track_window(
    rows: pd.DataFrame, *, start: int, length: int
) -> tuple[np.ndarray, ...]:
    position, velocity, heading, observed, mask = _empty_state_arrays(length)
    window = rows[(rows["timestep"] >= start) & (rows["timestep"] < start + length)]
    if window.empty:
        return position, velocity, heading, observed, mask
    indices = window["timestep"].to_numpy(dtype=np.int64) - start
    position[indices] = window[["position_x", "position_y"]].to_numpy(dtype=np.float32)
    velocity[indices] = window[["velocity_x", "velocity_y"]].to_numpy(dtype=np.float32)
    heading[indices] = window["heading"].to_numpy(dtype=np.float32)
    observed[indices] = window["observed"].to_numpy(dtype=bool)
    mask[indices] = True
    return position, velocity, heading, observed, mask


def build_focal_sample(
    scenario: AV2Scenario,
    *,
    split: str,
    history_steps: int = 50,
    future_steps: int = 60,
) -> TrajectorySample:
    """Align the focal track into causal history/future windows without interpolation."""

    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; received {split!r}")
    if history_steps <= 0 or future_steps <= 0:
        raise ValueError("history_steps and future_steps must both be positive.")
    required_steps = history_steps + future_steps
    if len(scenario.timestamps_ns) < required_steps:
        raise ScenarioValidationError(
            f"Scenario {scenario.scenario_id} has {len(scenario.timestamps_ns)} timestamps but "
            f"{required_steps} are required for {history_steps} history + {future_steps} future."
        )

    focal = scenario.tracks[
        scenario.tracks["track_id"].astype(str) == scenario.focal_track_id
    ].sort_values("timestep", kind="stable")
    invalid_history_flags = focal[(focal["timestep"] < history_steps) & ~focal["observed"]]
    invalid_future_flags = focal[
        (focal["timestep"] >= history_steps)
        & (focal["timestep"] < required_steps)
        & focal["observed"]
    ]
    if not invalid_history_flags.empty or not invalid_future_flags.empty:
        raise ScenarioValidationError(
            f"Scenario {scenario.scenario_id} focal observed flags do not match the "
            f"history/future boundary at timestep {history_steps}."
        )
    history = _align_track_window(focal, start=0, length=history_steps)
    future = _align_track_window(focal, start=history_steps, length=future_steps)
    history_position, history_velocity, history_heading, history_observed, history_mask = history
    future_position, future_velocity, future_heading, future_observed, future_mask = future
    if not history_mask.any():
        raise ScenarioValidationError(
            f"Scenario {scenario.scenario_id} focal track has no historical state."
        )
    if not future_mask.any():
        raise ScenarioValidationError(
            f"Scenario {scenario.scenario_id} focal track has no future label."
        )

    history_timesteps = np.arange(history_steps, dtype=np.int64)
    future_timesteps = np.arange(history_steps, required_steps, dtype=np.int64)
    if int(history_timesteps.max()) >= int(future_timesteps.min()):
        raise AssertionError("History/future timestep boundary is not causal.")
    last_valid_history = int(np.flatnonzero(history_mask)[-1])
    return TrajectorySample(
        scenario_id=scenario.scenario_id,
        split=split,
        focal_track_id=scenario.focal_track_id,
        city=scenario.city,
        history_timesteps=history_timesteps,
        future_timesteps=future_timesteps,
        timestamps_ns=scenario.timestamps_ns[:required_steps].astype(np.int64, copy=True),
        history_position=history_position,
        history_velocity=history_velocity,
        history_heading=history_heading,
        history_observed=history_observed,
        history_mask=history_mask,
        future_position=future_position,
        future_velocity=future_velocity,
        future_heading=future_heading,
        future_observed=future_observed,
        future_mask=future_mask,
        origin_xy=history_position[last_valid_history].copy(),
        reference_heading_rad=float(history_heading[last_valid_history]),
    )


def discover_scenario_paths(root: str | Path, split: str) -> list[Path]:
    """Discover scenario Parquet files under exactly one official split."""

    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; received {split!r}")
    split_root = Path(root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"AV2 split directory does not exist: {split_root}")
    paths = sorted(split_root.glob("*/scenario_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No scenario_*.parquet files found below: {split_root}")
    return paths


def select_scenario_paths(paths: Sequence[Path], count: int, seed: int) -> list[Path]:
    """Select a deterministic, ID-sorted random subset without replacement."""

    unique = sorted({Path(path) for path in paths}, key=lambda path: path.stem)
    if count <= 0:
        raise ValueError(f"Scene count must be positive; received {count}.")
    if count > len(unique):
        raise ValueError(f"Requested {count} scenes but only {len(unique)} are available.")
    selected = random.Random(seed).sample(unique, count)
    return sorted(selected, key=lambda path: path.stem)


_ARRAY_FIELDS = (
    "history_timesteps",
    "future_timesteps",
    "timestamps_ns",
    "history_position",
    "history_velocity",
    "history_heading",
    "history_observed",
    "history_mask",
    "future_position",
    "future_velocity",
    "future_heading",
    "future_observed",
    "future_mask",
    "origin_xy",
)


def write_cache_shards(
    samples: Sequence[TrajectorySample], output_dir: str | Path, *, shard_size: int
) -> list[dict[str, Any]]:
    """Write fixed-shape float32/bool cache shards; never overwrite existing output."""

    if not samples:
        raise ValueError("At least one sample is required to write cache shards.")
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive; received {shard_size}.")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Cache output already exists and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        chunk = list(samples[start : start + shard_size])
        filename = f"shard_{shard_index:04d}.npz"
        payload: dict[str, np.ndarray] = {
            "scenario_id": np.asarray([item.scenario_id for item in chunk]),
            "split": np.asarray([item.split for item in chunk]),
            "focal_track_id": np.asarray([item.focal_track_id for item in chunk]),
            "city": np.asarray([item.city for item in chunk]),
            "reference_heading_rad": np.asarray(
                [item.reference_heading_rad for item in chunk], dtype=np.float32
            ),
        }
        for field in _ARRAY_FIELDS:
            payload[field] = np.stack([np.asarray(getattr(item, field)) for item in chunk])
        np.savez_compressed(destination / filename, **payload)
        shards.append(
            {
                "file": filename,
                "sample_count": len(chunk),
                "scenario_ids": [item.scenario_id for item in chunk],
            }
        )
    return shards


def load_cache_shard(path: str | Path) -> list[TrajectorySample]:
    """Reload one cache shard without enabling pickle deserialization."""

    shard_path = Path(path)
    if not shard_path.is_file():
        raise FileNotFoundError(f"Cache shard does not exist: {shard_path}")
    samples: list[TrajectorySample] = []
    with np.load(shard_path, allow_pickle=False) as payload:
        count = len(payload["scenario_id"])
        for index in range(count):
            values = {field: payload[field][index].copy() for field in _ARRAY_FIELDS}
            samples.append(
                TrajectorySample(
                    scenario_id=str(payload["scenario_id"][index]),
                    split=str(payload["split"][index]),
                    focal_track_id=str(payload["focal_track_id"][index]),
                    city=str(payload["city"][index]),
                    reference_heading_rad=float(payload["reference_heading_rad"][index]),
                    **values,
                )
            )
    return samples


def write_json(path: str | Path, value: Any) -> Path:
    """Write reproducibility metadata as deterministic UTF-8 JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


class AV2MotionDataset:
    """Lazy reader for all Parquet scenarios in one official AV2 split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        history_steps: int = 50,
        future_steps: int = 60,
        paths: Sequence[Path] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.history_steps = history_steps
        self.future_steps = future_steps
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; received {split!r}")
        self.paths = list(paths) if paths is not None else discover_scenario_paths(root, split)

    def __iter__(self) -> Iterator[TrajectorySample]:
        for path in self.paths:
            yield build_focal_sample(
                read_av2_scenario(path),
                split=self.split,
                history_steps=self.history_steps,
                future_steps=self.future_steps,
            )
