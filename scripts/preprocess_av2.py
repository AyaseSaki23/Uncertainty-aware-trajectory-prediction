"""Build a reproducible AV2 focal-track subset, cache, metadata, and BEV artifact."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trajectory_prediction.config import load_yaml, require_keys
from trajectory_prediction.contracts import TrajectorySample
from trajectory_prediction.data.av2_dataset import (
    AV2MotionDataset,
    discover_scenario_paths,
    select_scenario_paths,
    write_cache_shards,
    write_json,
)
from trajectory_prediction.visualization.bev import plot_focal_sample_bev


def _require_nested(
    config: dict[str, object], section: str, keys: tuple[str, ...]
) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {section!r} must be a mapping.")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"Section {section!r} is missing keys: {', '.join(missing)}")
    return value


def _environment_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "matplotlib", "PyYAML")
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _summary(samples: list[TrajectorySample]) -> dict[str, object]:
    timestamp_deltas = np.concatenate(
        [np.diff(np.asarray(sample.timestamps_ns, dtype=np.int64)) / 1e9 for sample in samples]
    )
    return {
        "scenario_count": len(samples),
        "history_steps": int(len(samples[0].history_timesteps)),
        "future_steps": int(len(samples[0].future_timesteps)),
        "valid_history_states": int(
            sum(np.asarray(item.history_mask).sum() for item in samples)
        ),
        "valid_future_states": int(
            sum(np.asarray(item.future_mask).sum() for item in samples)
        ),
        "missing_history_states": int(
            sum((~np.asarray(item.history_mask)).sum() for item in samples)
        ),
        "missing_future_states": int(
            sum((~np.asarray(item.future_mask)).sum() for item in samples)
        ),
        "dt_seconds": {
            "minimum": float(timestamp_deltas.min()),
            "mean": float(timestamp_deltas.mean()),
            "maximum": float(timestamp_deltas.max()),
        },
        "array_dtypes": {"floating_point": "float32", "mask": "bool"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--output-dir",
        help="Override the non-existing output directory derived from the configuration.",
    )
    parser.add_argument(
        "--scene-count",
        type=int,
        help="Override subset.scene_count (use 1 for a first real-data smoke run).",
    )
    args = parser.parse_args()
    config = copy.deepcopy(load_yaml(args.config))
    require_keys(config, "dataset", "subset", "cache")
    dataset = _require_nested(
        config,
        "dataset",
        ("raw_root", "processed_root", "official_split", "history_steps", "future_steps"),
    )
    subset = _require_nested(config, "subset", ("scene_count", "sampling", "seed"))
    cache = _require_nested(config, "cache", ("shard_size", "float_dtype", "overwrite"))
    if subset["sampling"] != "seeded_random":
        raise ValueError("Only deterministic 'seeded_random' subset sampling is supported.")
    if cache["float_dtype"] != "float32":
        raise ValueError("Cache float_dtype must be 'float32'.")
    if bool(cache["overwrite"]):
        raise ValueError(
            "cache.overwrite must remain false; existing experiment output is immutable."
        )
    if args.check_config:
        print(f"Configuration is valid: {args.config}")
        return

    split = str(dataset["official_split"])
    count = args.scene_count if args.scene_count is not None else int(subset["scene_count"])
    if count <= 0:
        raise ValueError(f"Scene count must be positive; received {count}.")
    subset["scene_count"] = count
    seed = int(subset["seed"])
    default_name = f"subset_{count:03d}_seed_{seed}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(dataset["processed_root"]) / split / default_name
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_dir}"
        )

    discovered = discover_scenario_paths(dataset["raw_root"], split)
    selected = select_scenario_paths(discovered, count, seed)
    reader = AV2MotionDataset(
        dataset["raw_root"],
        split,
        history_steps=int(dataset["history_steps"]),
        future_steps=int(dataset["future_steps"]),
        paths=selected,
    )
    samples = list(reader)
    shards = write_cache_shards(samples, output_dir / "cache", shard_size=int(cache["shard_size"]))

    manifest = {
        "format_version": 1,
        "dataset": "argoverse2_motion_forecasting",
        "split": split,
        "sampling": {
            "method": "seeded_random_without_replacement",
            "seed": seed,
            "requested_scene_count": count,
            "available_scene_count": len(discovered),
            "ordering": "scenario_id_ascending_after_sampling",
        },
        "window": {
            "history_steps": int(dataset["history_steps"]),
            "future_steps": int(dataset["future_steps"]),
            "timestamp_source": "scenario start/end/count, reconstructed in integer nanoseconds",
        },
        "scenario_ids": [sample.scenario_id for sample in samples],
        "source_files": [
            str(path.relative_to(Path(dataset["raw_root"]) / split)) for path in selected
        ],
        "shards": shards,
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    summary_path = write_json(output_dir / "data_summary.json", _summary(samples))
    versions_path = write_json(output_dir / "environment_versions.json", _environment_versions())
    snapshot_path = output_dir / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    plot_path = plot_focal_sample_bev(
        samples[0], output_dir / "artifacts" / f"bev_{samples[0].scenario_id}.png"
    )

    print(f"Processed {len(samples)} AV2 {split} scenarios.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Environment: {versions_path}")
    print(f"Config snapshot: {snapshot_path}")
    print(f"Fixed-scene BEV: {plot_path}")
    print(f"Python executable: {sys.executable}")


if __name__ == "__main__":
    main()
