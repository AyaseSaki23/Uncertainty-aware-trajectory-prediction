"""Generate validated CSV and Markdown tables from saved CV evaluation runs."""

from __future__ import annotations

import argparse

from trajectory_prediction.reporting.tables import generate_cv_results_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        help="CV evaluation run directory; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = generate_cv_results_tables(args.run_dir, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
