"""Logical reproducibility checks for two saved CV evaluation runs."""

from __future__ import annotations

import csv
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ReproducibilityError(ValueError):
    """Raised when two evaluation runs differ in logical content."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproducibility input does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON reproducibility input: {path}") from exc


def _compare_values(path: str, left: Any, right: Any, *, atol: float) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ReproducibilityError(
                f"{path} keys differ: {sorted(left)} vs {sorted(right)}."
            )
        for key in sorted(left):
            _compare_values(f"{path}.{key}", left[key], right[key], atol=atol)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ReproducibilityError(
                f"{path} lengths differ: {len(left)} vs {len(right)}."
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _compare_values(f"{path}[{index}]", left_item, right_item, atol=atol)
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ReproducibilityError(f"{path} differs: {left!r} vs {right!r}.")
        return
    if isinstance(left, Real) and isinstance(right, Real):
        if not np.isfinite([float(left), float(right)]).all():
            raise ReproducibilityError(f"{path} contains NaN or Inf.")
        if not np.isclose(float(left), float(right), rtol=0.0, atol=atol):
            raise ReproducibilityError(f"{path} differs: {left!r} vs {right!r}.")
        return
    if left != right:
        raise ReproducibilityError(f"{path} differs: {left!r} vs {right!r}.")


def _normalized_run_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("run_config.json root must be an object.")
    normalized = dict(value)
    normalized.pop("output_dir", None)
    normalized.pop("manifest", None)
    return normalized


def _resolve_manifest(run_dir: Path, config: Mapping[str, Any]) -> Path:
    raw_path = config.get("manifest")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"run_config.json has no valid manifest path: {run_dir}")
    candidate = Path(raw_path)
    project_root = Path(__file__).resolve().parents[3]
    for choice in (candidate, run_dir / candidate, project_root / candidate):
        if choice.is_file():
            return choice
    raise FileNotFoundError(f"Recorded manifest cannot be resolved for run: {run_dir}")


def _verified_manifest_hash(
    run_dir: Path, config: Mapping[str, Any], summary: Mapping[str, Any]
) -> str:
    manifest = _resolve_manifest(run_dir, config)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for source, recorded in (
        ("run_config", config.get("manifest_sha256")),
        ("summary", summary.get("manifest_sha256")),
    ):
        if recorded != digest:
            raise ReproducibilityError(
                f"{run_dir} {source} manifest_sha256 does not match current manifest."
            )
    return digest


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproducibility input does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise ValueError(f"CSV reproducibility input must have a header and rows: {path}")
    return fields, rows


def _compare_csv(
    name: str,
    left_path: Path,
    right_path: Path,
    *,
    numeric_fields: Sequence[str],
    atol: float,
) -> int:
    left_fields, left_rows = _load_csv(left_path)
    right_fields, right_rows = _load_csv(right_path)
    if left_fields != right_fields:
        raise ReproducibilityError(
            f"{name} headers differ: {left_fields} vs {right_fields}."
        )
    if len(left_rows) != len(right_rows):
        raise ReproducibilityError(
            f"{name} row counts differ: {len(left_rows)} vs {len(right_rows)}."
        )
    numeric = set(numeric_fields)
    for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
        for field in left_fields:
            path = f"{name}[{row_index}].{field}"
            if field in numeric:
                try:
                    left_value = float(left_row[field])
                    right_value = float(right_row[field])
                except ValueError as exc:
                    raise ReproducibilityError(f"{path} is not numeric.") from exc
                _compare_values(path, left_value, right_value, atol=atol)
            elif left_row[field] != right_row[field]:
                raise ReproducibilityError(
                    f"{path} differs: {left_row[field]!r} vs {right_row[field]!r}."
                )
    return len(left_rows)


def _compare_prediction_arrays(
    left_path: Path, right_path: Path, *, atol: float
) -> tuple[dict[str, dict[str, object]], int]:
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError("Both runs must contain predictions.npz.")
    details: dict[str, dict[str, object]] = {}
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            raise ReproducibilityError(
                f"Prediction array keys differ: {sorted(left.files)} vs {sorted(right.files)}."
            )
        for key in sorted(left.files):
            left_array = left[key]
            right_array = right[key]
            if left_array.shape != right_array.shape:
                raise ReproducibilityError(
                    f"predictions.{key} shapes differ: "
                    f"{left_array.shape} vs {right_array.shape}."
                )
            if left_array.dtype != right_array.dtype:
                raise ReproducibilityError(
                    f"predictions.{key} dtypes differ: "
                    f"{left_array.dtype} vs {right_array.dtype}."
                )
            detail: dict[str, object] = {
                "shape": list(left_array.shape),
                "dtype": str(left_array.dtype),
            }
            if left_array.dtype.kind == "f":
                if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
                    raise ReproducibilityError(f"predictions.{key} contains NaN or Inf.")
                difference = np.abs(left_array - right_array)
                maximum = float(difference.max(initial=0.0))
                detail["max_abs_difference"] = maximum
                if maximum > atol:
                    raise ReproducibilityError(
                        f"predictions.{key} max absolute difference {maximum} "
                        f"exceeds tolerance {atol}."
                    )
            elif not np.array_equal(left_array, right_array):
                raise ReproducibilityError(f"predictions.{key} values differ.")
            details[key] = detail
        sample_count = int(left["scenario_id"].shape[0])
    return details, sample_count


def compare_cv_runs(
    run_a: str | Path,
    run_b: str | Path,
    *,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Compare two CV runs by logical JSON, CSV, manifest, and array content."""

    left = Path(run_a)
    right = Path(run_b)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be finite and non-negative; received {atol!r}.")
    missing = [str(path) for path in (left, right) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"CV run directories do not exist: {missing}")

    left_config = _load_json(left / "run_config.json")
    right_config = _load_json(right / "run_config.json")
    left_summary = _load_json(left / "summary.json")
    right_summary = _load_json(right / "summary.json")
    if not isinstance(left_summary, dict) or not isinstance(right_summary, dict):
        raise ValueError("summary.json roots must be objects.")
    _compare_values(
        "run_config",
        _normalized_run_config(left_config),
        _normalized_run_config(right_config),
        atol=atol,
    )
    _compare_values("summary", left_summary, right_summary, atol=atol)
    left_manifest_hash = _verified_manifest_hash(left, left_config, left_summary)
    right_manifest_hash = _verified_manifest_hash(right, right_config, right_summary)
    if left_manifest_hash != right_manifest_hash:
        raise ReproducibilityError(
            f"Manifest hashes differ: {left_manifest_hash} vs {right_manifest_hash}."
        )

    metric_count = _compare_csv(
        "per_scene_metrics",
        left / "per_scene_metrics.csv",
        right / "per_scene_metrics.csv",
        numeric_fields=(
            "valid_future_steps",
            "last_valid_history_index",
            "ADE",
            "FDE",
            "minADE",
            "minFDE",
            "miss_threshold_m",
        ),
        atol=atol,
    )
    index_count = _compare_csv(
        "prediction_index",
        left / "prediction_index.csv",
        right / "prediction_index.csv",
        numeric_fields=("prediction_row", "future_steps", "valid_future_steps"),
        atol=atol,
    )
    _compare_values(
        "failures",
        _load_json(left / "failures.json"),
        _load_json(right / "failures.json"),
        atol=atol,
    )
    _compare_values(
        "environment_versions",
        _load_json(left / "environment_versions.json"),
        _load_json(right / "environment_versions.json"),
        atol=atol,
    )
    array_details, prediction_count = _compare_prediction_arrays(
        left / "predictions.npz", right / "predictions.npz", atol=atol
    )
    if len({metric_count, index_count, prediction_count}) != 1:
        raise ReproducibilityError(
            "Metrics, prediction index, and prediction arrays have different sample counts."
        )
    return {
        "format_version": 1,
        "matched": True,
        "run_a": str(left),
        "run_b": str(right),
        "absolute_tolerance": atol,
        "manifest_sha256": left_manifest_hash,
        "sample_count": metric_count,
        "prediction_arrays": array_details,
        "ignored_container_bytes": True,
        "ignored_run_config_fields": ["manifest", "output_dir"],
    }
