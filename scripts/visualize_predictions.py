"""Visualize saved scenario predictions and uncertainty diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if args.check_only:
        print(f"Run directory argument accepted: {run_dir}")
        return
    raise NotImplementedError("Prediction visualization is implemented with saved result schemas.")


if __name__ == "__main__":
    main()
