"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail with an actionable error for invalid input."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return value


def require_keys(config: dict[str, Any], *keys: str) -> None:
    """Require top-level keys in a parsed configuration."""

    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
