"""Evaluate a checkpoint using aligned clean or degraded scenarios."""

from __future__ import annotations

import argparse

from trajectory_prediction.config import load_yaml, require_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    require_keys(config, "experiment", "data_config", "model_config", "input")
    if args.check_config:
        print(f"Configuration is valid: {args.config}")
        return
    raise NotImplementedError("Evaluation will be implemented with validated metrics.")


if __name__ == "__main__":
    main()
