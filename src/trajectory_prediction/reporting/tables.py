"""Generate paper-ready CV tables from saved structured evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_SUMMARY_KEYS = (
    "model",
    "split",
    "sample_count",
    "failed_scene_count",
    "valid_future_state_count",
    "ADE",
    "FDE",
    "minADE",
    "minFDE",
    "MissRate",
    "miss_threshold_m",
    "git_commit",
    "manifest_sha256",
)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"CV summary does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"CV summary root must be a JSON object: {path}")
    missing = [key for key in _SUMMARY_KEYS if key not in value]
    if missing:
        raise ValueError(f"CV summary is missing keys: {', '.join(missing)}")
    if value["model"] != "cv" or value["split"] != "val":
        raise ValueError("Result tables require model='cv' on the official val split.")
    if not isinstance(value["sample_count"], int) or value["sample_count"] <= 0:
        raise ValueError("CV summary sample_count must be a positive integer.")
    return value


def _parse_miss(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Per-scene miss must be True or False; received {value!r}.")


def _recompute_metrics(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "per_scene_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Per-scene metrics do not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Per-scene metrics contain no rows: {path}")
    try:
        ade = np.asarray([float(row["ADE"]) for row in rows], dtype=np.float64)
        fde = np.asarray([float(row["FDE"]) for row in rows], dtype=np.float64)
        min_ade = np.asarray([float(row["minADE"]) for row in rows], dtype=np.float64)
        min_fde = np.asarray([float(row["minFDE"]) for row in rows], dtype=np.float64)
        miss = np.asarray([_parse_miss(row["miss"]) for row in rows], dtype=bool)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid per-scene metric values in {path}.") from exc
    if not np.isfinite(np.concatenate((ade, fde, min_ade, min_fde))).all():
        raise ValueError(f"Per-scene metrics contain NaN or Inf: {path}")
    return {
        "sample_count": len(rows),
        "ADE": float(np.mean(ade, dtype=np.float64)),
        "FDE": float(np.mean(fde, dtype=np.float64)),
        "minADE": float(np.mean(min_ade, dtype=np.float64)),
        "minFDE": float(np.mean(min_fde, dtype=np.float64)),
        "MissRate": float(np.mean(miss, dtype=np.float64)),
    }


def _validated_table_row(run_dir: Path) -> dict[str, object]:
    summary = _load_summary(run_dir)
    recomputed = _recompute_metrics(run_dir)
    if recomputed["sample_count"] != summary["sample_count"]:
        raise ValueError(
            f"summary/per-scene sample_count mismatch in {run_dir}: "
            f"{summary['sample_count']} vs {recomputed['sample_count']}."
        )
    for key in ("ADE", "FDE", "minADE", "minFDE", "MissRate"):
        expected = float(summary[key])
        actual = float(recomputed[key])
        if not np.isfinite(expected) or not np.isclose(expected, actual, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"summary/per-scene {key} mismatch in {run_dir}: "
                f"{expected} vs {actual}."
            )
    return {
        "evaluation_scope": "fixed_validation_subset",
        "model": summary["model"],
        "split": summary["split"],
        "scene_count": summary["sample_count"],
        "failed_scene_count": summary["failed_scene_count"],
        "valid_future_state_count": summary["valid_future_state_count"],
        "ADE_m": float(summary["ADE"]),
        "FDE_m": float(summary["FDE"]),
        "MissRate": float(summary["MissRate"]),
        "miss_threshold_m": float(summary["miss_threshold_m"]),
        "git_commit": summary["git_commit"],
        "manifest_sha256": summary["manifest_sha256"],
        "source_run_dir": str(run_dir),
    }


def _markdown_table(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Constant Velocity Baseline — Fixed AV2 Validation Subset",
        "",
        "该表仅代表固定 validation 子集，不是 AV2 官方完整测试集结果。距离单位为米。",
        "",
        "| Model | Split | Scenes | Failed | ADE (m) | FDE (m) | Miss Rate | "
        "Threshold (m) | Git commit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['scene_count']} | "
            f"{row['failed_scene_count']} | {float(row['ADE_m']):.6f} | "
            f"{float(row['FDE_m']):.6f} | {float(row['MissRate']):.6f} | "
            f"{float(row['miss_threshold_m']):.3f} | `{row['git_commit']}` |"
        )
    return "\n".join(lines) + "\n"


def generate_cv_results_tables(
    run_dirs: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Path]:
    """Validate CV runs and write one CSV plus one Markdown result table."""

    if not run_dirs:
        raise ValueError("At least one CV run directory is required.")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Table output exists and will not be overwritten: {destination}"
        )
    normalized = [Path(value) for value in run_dirs]
    if any(not path.is_dir() for path in normalized):
        missing = [str(path) for path in normalized if not path.is_dir()]
        raise FileNotFoundError(f"CV run directories do not exist: {missing}")
    rows = [_validated_table_row(path) for path in normalized]

    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / "cv_baseline_results.csv"
    markdown_path = destination / "cv_baseline_results.md"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown_table(rows), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}
