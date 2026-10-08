"""Load saved CV results and generate reproducible review visualizations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajectory_prediction.data.av2_dataset import write_json
from trajectory_prediction.evaluation.cv import load_samples_from_manifest
from trajectory_prediction.geometry.coordinates import wrap_angle
from trajectory_prediction.visualization.bev import plot_prediction_comparison


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CV result file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CV result file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV result file contains no rows: {path}")
    return rows


def _resolve_input_path(run_dir: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("run_config manifest must be a non-empty path string.")
    candidate = Path(raw_path)
    project_root = Path(__file__).resolve().parents[3]
    choices = [candidate, run_dir / candidate, project_root / candidate]
    for choice in choices:
        if choice.is_file():
            return choice
    raise FileNotFoundError(
        f"Manifest recorded by run_config cannot be found; checked: {choices}"
    )


def _resolve_data_root(raw_path: Any) -> Path | None:
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("run_config data_root must be null or a non-empty path string.")
    candidate = Path(raw_path)
    if candidate.is_dir():
        return candidate
    project_relative = Path(__file__).resolve().parents[3] / candidate
    return project_relative if project_relative.is_dir() else candidate


def _heading_change_rad(sample: Any) -> float:
    history_mask = np.asarray(sample.history_mask)
    future_mask = np.asarray(sample.future_mask)
    history_index = int(np.flatnonzero(history_mask)[-1])
    future_index = int(np.flatnonzero(future_mask)[-1])
    change = wrap_angle(
        np.asarray(sample.future_heading)[future_index]
        - np.asarray(sample.history_heading)[history_index]
    )
    return abs(float(change))


def _case_metadata(
    metric_rows: Sequence[dict[str, str]],
    samples_by_id: dict[str, Any],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for row in metric_rows:
        scenario_id = row.get("scenario_id", "")
        if scenario_id not in samples_by_id:
            raise ValueError(f"Metrics reference unknown scenario_id {scenario_id!r}.")
        sample = samples_by_id[scenario_id]
        try:
            ade = float(row["ADE"])
            fde = float(row["FDE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ADE/FDE for scenario {scenario_id!r}.") from exc
        if not np.isfinite([ade, fde]).all():
            raise ValueError(f"Non-finite ADE/FDE for scenario {scenario_id!r}.")
        cases.append(
            {
                "scenario_id": scenario_id,
                "ADE": ade,
                "FDE": fde,
                "heading_change_rad": _heading_change_rad(sample),
                "missing_history_steps": int((~np.asarray(sample.history_mask)).sum()),
                "missing_future_steps": int((~np.asarray(sample.future_mask)).sum()),
            }
        )
    return cases


def _select_review_cases(
    cases: Sequence[dict[str, object]], max_scenes: int
) -> list[dict[str, object]]:
    if max_scenes <= 0:
        raise ValueError(f"max_scenes must be positive; received {max_scenes}.")
    rankings = (
        ("straight", sorted(cases, key=lambda item: float(item["heading_change_rad"]))),
        (
            "turn",
            sorted(cases, key=lambda item: float(item["heading_change_rad"]), reverse=True),
        ),
        ("cv_failure", sorted(cases, key=lambda item: float(item["FDE"]), reverse=True)),
    )
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for category, ranking in rankings:
        if len(selected) >= max_scenes:
            break
        candidate = next(
            (item for item in ranking if str(item["scenario_id"]) not in used),
            None,
        )
        if candidate is not None:
            chosen = dict(candidate)
            chosen["review_category"] = category
            selected.append(chosen)
            used.add(str(chosen["scenario_id"]))
    return selected


def _review_checklist(cases: Sequence[dict[str, object]]) -> str:
    lines = [
        "# CV 场景人工审核清单",
        "",
        "逐图确认全局/局部坐标方向、历史未来边界、等比例坐标轴和 mask 缺口。",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['review_category']}: `{case['scenario_id']}`",
                "",
                f"- ADE: {float(case['ADE']):.6f} m",
                f"- FDE: {float(case['FDE']):.6f} m",
                f"- 航向变化: {float(case['heading_change_rad']):.6f} rad",
                f"- 缺失历史/未来: {case['missing_history_steps']} / "
                f"{case['missing_future_steps']}",
                "- [ ] 全局与局部轨迹方向一致",
                "- [ ] 历史/未来边界正确",
                "- [ ] 缺失状态没有被跨段连线",
                "- [ ] CV 成功或失败形态与 ADE/FDE 一致",
                "- 审核备注：",
                "",
            ]
        )
    return "\n".join(lines)


def generate_review_visualizations(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_ids: Sequence[str] | None = None,
    selection: str = "review",
    max_scenes: int = 3,
) -> dict[str, object]:
    """Generate global/local CV plots and a review manifest from one run."""

    run = Path(run_dir)
    destination = Path(output_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"CV run directory does not exist: {run}")
    if destination.exists():
        raise FileExistsError(
            f"Visualization output exists and will not be overwritten: {destination}"
        )
    if selection not in {"review", "all"}:
        raise ValueError("selection must be either 'review' or 'all'.")

    run_config = _load_json_object(run / "run_config.json")
    manifest_path = _resolve_input_path(run, run_config.get("manifest"))
    _, samples = load_samples_from_manifest(
        manifest_path,
        data_root=_resolve_data_root(run_config.get("data_root")),
    )
    samples_by_id = {sample.scenario_id: sample for sample in samples}
    metric_rows = _load_csv_rows(run / "per_scene_metrics.csv")
    index_rows = _load_csv_rows(run / "prediction_index.csv")
    metrics_by_id = {row["scenario_id"]: row for row in metric_rows}
    if len(metrics_by_id) != len(metric_rows):
        raise ValueError("Per-scene metrics contain duplicate scenario IDs.")
    try:
        prediction_rows = [int(row["prediction_row"]) for row in index_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prediction index contains invalid row numbers.") from exc
    if prediction_rows != list(range(len(index_rows))):
        raise ValueError("Prediction index rows must be contiguous and ordered from zero.")
    index_by_id = {
        row["scenario_id"]: prediction_rows[index]
        for index, row in enumerate(index_rows)
    }
    if len(index_by_id) != len(index_rows):
        raise ValueError("Prediction index contains duplicate scenario IDs.")
    if set(metrics_by_id) != set(index_by_id):
        raise ValueError("Metric and prediction-index scenario IDs do not match.")
    cases = _case_metadata(metric_rows, samples_by_id)

    if scenario_ids:
        unknown = [item for item in scenario_ids if item not in metrics_by_id]
        if unknown:
            raise ValueError(f"Requested scenario IDs are not in the run: {unknown}")
        cases_by_id = {str(item["scenario_id"]): item for item in cases}
        selected = []
        for scenario_id in scenario_ids:
            chosen = dict(cases_by_id[scenario_id])
            chosen["review_category"] = "manual"
            selected.append(chosen)
    elif selection == "all":
        selected = []
        for case in cases:
            chosen = dict(case)
            chosen["review_category"] = "all"
            selected.append(chosen)
    else:
        selected = _select_review_cases(cases, max_scenes=max_scenes)

    predictions_path = run / "predictions.npz"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Required CV result file does not exist: {predictions_path}")
    with np.load(predictions_path, allow_pickle=False) as payload:
        payload_ids = [str(value) for value in payload["scenario_id"]]
        if payload_ids != [row["scenario_id"] for row in index_rows]:
            raise ValueError("Prediction payload order does not match prediction_index.csv.")
        local_predictions = payload["local_prediction_m"].copy()
        global_predictions = payload["global_prediction_m"].copy()
    if local_predictions.shape[0] != len(index_rows) or global_predictions.shape[0] != len(
        index_rows
    ):
        raise ValueError("Prediction payload batch size does not match prediction index.")

    destination.mkdir(parents=True, exist_ok=False)
    generated: list[str] = []
    for order, case in enumerate(selected, start=1):
        scenario_id = str(case["scenario_id"])
        prediction_row = index_by_id[scenario_id]
        filename = f"{order:02d}_{case['review_category']}_{scenario_id}.png"
        output_path = destination / filename
        plot_prediction_comparison(
            samples_by_id[scenario_id],
            local_predictions[prediction_row],
            global_predictions[prediction_row],
            metrics_by_id[scenario_id],
            output_path,
        )
        case["figure"] = filename
        generated.append(filename)

    review_manifest = {
        "format_version": 1,
        "source_run_dir": str(run),
        "selection": "manual" if scenario_ids else selection,
        "generated_scene_count": len(selected),
        "cases": selected,
    }
    write_json(destination / "review_manifest.json", review_manifest)
    (destination / "review_checklist.md").write_text(
        _review_checklist(selected) + "\n", encoding="utf-8"
    )
    return {**review_manifest, "generated_files": generated}
