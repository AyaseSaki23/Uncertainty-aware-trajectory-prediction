"""Reproducible Constant Velocity evaluation on a fixed AV2 validation manifest."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.data.av2_dataset import (
    build_focal_sample,
    load_cache_shard,
    read_av2_scenario,
    write_json,
)
from trajectory_prediction.geometry.coordinates import (
    local_to_global,
    trajectory_sample_to_local,
)
from trajectory_prediction.metrics.forecasting import (
    compute_per_sample_forecasting_metrics,
)
from trajectory_prediction.models.constant_velocity import (
    predict_constant_velocity_sample,
)


@dataclass(frozen=True)
class CVEvaluationResult:
    """In-memory per-scene CV metrics and aligned prediction arrays."""

    rows: tuple[dict[str, object], ...]
    failures: tuple[dict[str, str], ...]
    scenario_ids: tuple[str, ...]
    local_predictions_m: tuple[np.ndarray, ...]
    global_predictions_m: tuple[np.ndarray, ...]
    ground_truth_global_m: tuple[np.ndarray, ...]
    future_masks: tuple[np.ndarray, ...]
    future_timestamps_ns: tuple[np.ndarray, ...]
    summary: dict[str, object]


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a fixed AV2 validation manifest JSON file."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Evaluation manifest root must be a JSON object.")
    required = ("format_version", "dataset", "split", "window", "scenario_ids")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Evaluation manifest is missing keys: {', '.join(missing)}")
    if value["format_version"] != 1:
        raise ValueError(
            f"Unsupported evaluation manifest format_version: {value['format_version']!r}."
        )
    if value["dataset"] != "argoverse2_motion_forecasting":
        raise ValueError("Evaluation manifest dataset must be argoverse2_motion_forecasting.")
    if value["split"] != "val":
        raise ValueError(
            f"CV evaluation requires the official val split; received {value['split']!r}."
        )
    window = value["window"]
    if not isinstance(window, dict):
        raise ValueError("Evaluation manifest window must be a JSON object.")
    for key in ("history_steps", "future_steps"):
        if not isinstance(window.get(key), int) or int(window[key]) <= 0:
            raise ValueError(f"Evaluation manifest window.{key} must be a positive integer.")
    scenario_ids = value["scenario_ids"]
    if (
        not isinstance(scenario_ids, list)
        or not scenario_ids
        or not all(isinstance(item, str) and item for item in scenario_ids)
    ):
        raise ValueError("Evaluation manifest scenario_ids must be a non-empty string list.")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Evaluation manifest scenario_ids must be unique.")
    return value


def _cache_paths(manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        return []
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or not isinstance(shard.get("file"), str):
            raise ValueError(f"Evaluation manifest shards[{index}] needs a string file field.")
        filename = Path(shard["file"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError(f"Unsafe cache shard path in manifest: {filename}")
        paths.append(manifest_path.parent / "cache" / filename)
    return paths


def _raw_paths(
    manifest: dict[str, Any], data_root: str | Path | None
) -> list[Path]:
    if data_root is None:
        raise FileNotFoundError(
            "Manifest cache shards are unavailable. Pass --data-root pointing to the "
            "AV2 root containing the val split."
        )
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("Evaluation manifest needs source_files for raw-data fallback.")
    split_root = Path(data_root) / str(manifest["split"])
    paths: list[Path] = []
    for value in source_files:
        if not isinstance(value, str):
            raise ValueError("Evaluation manifest source_files entries must be strings.")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source file path in manifest: {relative}")
        paths.append(split_root / relative)
    return paths


def load_samples_from_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[TrajectorySample]]:
    """Load manifest-ordered global samples from cache or explicit AV2 raw root."""

    path = Path(manifest_path)
    manifest = load_evaluation_manifest(path)
    cache_paths = _cache_paths(path, manifest)
    if cache_paths and all(item.is_file() for item in cache_paths):
        samples = [sample for shard in cache_paths for sample in load_cache_shard(shard)]
    else:
        raw_paths = _raw_paths(manifest, data_root)
        window = manifest["window"]
        samples = [
            build_focal_sample(
                read_av2_scenario(source),
                split="val",
                history_steps=int(window["history_steps"]),
                future_steps=int(window["future_steps"]),
            )
            for source in raw_paths
        ]

    expected_ids = list(manifest["scenario_ids"])
    actual_ids = [sample.scenario_id for sample in samples]
    if actual_ids != expected_ids:
        raise ValueError(
            "Loaded scenario order does not match manifest scenario_ids; "
            f"expected {expected_ids}, received {actual_ids}."
        )
    if any(sample.split != "val" for sample in samples):
        raise ValueError("Every evaluation sample must belong to the official val split.")
    if any(sample.coordinate_frame != "global" for sample in samples):
        raise ValueError("Evaluation manifest caches must contain global-coordinate samples.")
    return manifest, samples


def _validate_threshold(miss_threshold_m: Any) -> float:
    threshold = np.asarray(miss_threshold_m)
    if threshold.shape != () or threshold.dtype.kind not in "iuf":
        raise ValueError("miss_threshold_m must be one finite positive scalar in metres.")
    value = float(threshold)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("miss_threshold_m must be one finite positive scalar in metres.")
    return value


def evaluate_cv_samples(
    samples: Sequence[TrajectorySample],
    *,
    miss_threshold_m: float = 2.0,
) -> CVEvaluationResult:
    """Localize global samples, run causal CV, and compute per-scene metrics."""

    if not samples:
        raise ValueError("At least one validation sample is required for CV evaluation.")
    threshold = _validate_threshold(miss_threshold_m)
    scenario_ids = [sample.scenario_id for sample in samples]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("CV evaluation sample scenario_id values must be unique.")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    successful_ids: list[str] = []
    local_predictions: list[np.ndarray] = []
    global_predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    future_masks: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []
    expected_future_steps: int | None = None

    for sample in samples:
        try:
            if sample.split != "val":
                raise ValueError("sample split must be 'val'")
            if sample.coordinate_frame != "global":
                raise ValueError("sample coordinate_frame must be 'global'")
            local_sample = trajectory_sample_to_local(sample)
            prediction_local = predict_constant_velocity_sample(local_sample)
            future_steps = int(prediction_local.shape[0])
            if expected_future_steps is None:
                expected_future_steps = future_steps
            elif future_steps != expected_future_steps:
                raise ValueError(
                    f"future length {future_steps} differs from {expected_future_steps}"
                )
            target_local = np.asarray(local_sample.future_position, dtype=np.float64)
            mask = np.asarray(local_sample.future_mask)
            metrics = compute_per_sample_forecasting_metrics(
                prediction_local[None, :, :],
                target_local[None, :, :],
                mask[None, :],
                miss_threshold_m=threshold,
            )
            prediction_global = local_to_global(
                prediction_local,
                local_sample.origin_xy,
                local_sample.reference_heading_rad,
            ).astype(np.float64, copy=False)
            target_global = np.asarray(sample.future_position, dtype=np.float64)
            timestep = np.asarray(sample.future_timesteps, dtype=np.int64)
            timestamps = np.asarray(sample.timestamps_ns)[timestep].astype(
                np.int64, copy=True
            )
            last_valid_history = int(np.flatnonzero(sample.history_mask)[-1])
            ade = float(metrics["ade_by_mode"][0, 0])
            fde = float(metrics["fde_by_mode"][0, 0])
            min_ade = float(metrics["min_ade"][0])
            min_fde = float(metrics["min_fde"][0])
            miss = bool(metrics["miss"][0])
            rows.append(
                {
                    "scenario_id": sample.scenario_id,
                    "split": sample.split,
                    "model": "cv",
                    "valid_future_steps": int(mask.sum()),
                    "last_valid_history_index": last_valid_history,
                    "ADE": ade,
                    "FDE": fde,
                    "minADE": min_ade,
                    "minFDE": min_fde,
                    "miss": miss,
                    "miss_threshold_m": threshold,
                }
            )
            successful_ids.append(sample.scenario_id)
            local_predictions.append(prediction_local.astype(np.float64, copy=False))
            global_predictions.append(prediction_global)
            ground_truth.append(target_global)
            future_masks.append(mask.copy())
            future_timestamps.append(timestamps)
        except (IndexError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "scenario_id": sample.scenario_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    if not rows:
        raise ValueError(f"All {len(samples)} CV evaluation samples failed validation.")
    summary: dict[str, object] = {
        "format_version": 1,
        "model": "cv",
        "split": "val",
        "requested_scene_count": len(samples),
        "sample_count": len(rows),
        "failed_scene_count": len(failures),
        "mode_count": 1,
        "valid_future_state_count": int(
            sum(int(row["valid_future_steps"]) for row in rows)
        ),
        "ADE": float(np.mean([float(row["ADE"]) for row in rows], dtype=np.float64)),
        "FDE": float(np.mean([float(row["FDE"]) for row in rows], dtype=np.float64)),
        "minADE": float(
            np.mean([float(row["minADE"]) for row in rows], dtype=np.float64)
        ),
        "minFDE": float(
            np.mean([float(row["minFDE"]) for row in rows], dtype=np.float64)
        ),
        "MissRate": float(
            np.mean([bool(row["miss"]) for row in rows], dtype=np.float64)
        ),
        "miss_threshold_m": threshold,
    }
    return CVEvaluationResult(
        rows=tuple(rows),
        failures=tuple(failures),
        scenario_ids=tuple(successful_ids),
        local_predictions_m=tuple(local_predictions),
        global_predictions_m=tuple(global_predictions),
        ground_truth_global_m=tuple(ground_truth),
        future_masks=tuple(future_masks),
        future_timestamps_ns=tuple(future_timestamps),
        summary=summary,
    )


def _environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "pyarrow", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV rows: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_result_files(
    result: CVEvaluationResult,
    output_dir: Path,
    *,
    run_config: dict[str, object],
    summary: dict[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "per_scene_metrics.csv"
    index_path = output_dir / "prediction_index.csv"
    predictions_path = output_dir / "predictions.npz"
    _write_csv(metrics_path, result.rows)
    index_rows = [
        {
            "prediction_row": index,
            "scenario_id": scenario_id,
            "predictions_file": predictions_path.name,
            "future_steps": int(result.local_predictions_m[index].shape[0]),
            "valid_future_steps": int(result.future_masks[index].sum()),
        }
        for index, scenario_id in enumerate(result.scenario_ids)
    ]
    _write_csv(index_path, index_rows)
    np.savez_compressed(
        predictions_path,
        scenario_id=np.asarray(result.scenario_ids),
        local_prediction_m=np.stack(result.local_predictions_m),
        global_prediction_m=np.stack(result.global_predictions_m),
        ground_truth_global_m=np.stack(result.ground_truth_global_m),
        future_mask=np.stack(result.future_masks),
        future_timestamps_ns=np.stack(result.future_timestamps_ns),
    )
    paths = {
        "summary": write_json(output_dir / "summary.json", summary),
        "metrics": metrics_path,
        "prediction_index": index_path,
        "predictions": predictions_path,
        "failures": write_json(output_dir / "failures.json", list(result.failures)),
        "run_config": write_json(output_dir / "run_config.json", run_config),
        "environment": write_json(
            output_dir / "environment_versions.json", _environment_versions()
        ),
    }
    return paths


def run_cv_evaluation(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    miss_threshold_m: float = 2.0,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Run fixed-manifest CV evaluation and atomically refuse existing output."""

    manifest_file = Path(manifest_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Evaluation output already exists and will not be overwritten: {destination}"
        )
    manifest, samples = load_samples_from_manifest(
        manifest_file, data_root=data_root
    )
    result = evaluate_cv_samples(samples, miss_threshold_m=miss_threshold_m)
    commit = _git_commit()
    manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    summary = {
        **result.summary,
        "git_commit": commit,
        "manifest_sha256": manifest_sha256,
        "manifest_scenario_count": len(manifest["scenario_ids"]),
    }
    run_config: dict[str, object] = {
        "format_version": 1,
        "model": "cv",
        "manifest": str(manifest_file),
        "data_root": None if data_root is None else str(data_root),
        "output_dir": str(destination),
        "miss_threshold_m": float(summary["miss_threshold_m"]),
        "git_commit": commit,
        "manifest_sha256": manifest_sha256,
    }
    paths = _write_result_files(
        result,
        destination,
        run_config=run_config,
        summary=summary,
    )
    return summary, paths
