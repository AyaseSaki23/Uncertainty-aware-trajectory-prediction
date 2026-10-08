"""Compare two CV evaluation runs by logical content, not NPZ container bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_prediction.data.av2_dataset import write_json
from trajectory_prediction.evaluation.reproducibility import compare_cv_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output", help="Optional non-existing JSON report path.")
    args = parser.parse_args()

    report = compare_cv_runs(args.run_a, args.run_b, atol=args.atol)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(
                f"Comparison report exists and will not be overwritten: {output}"
            )
        write_json(output, report)
        print(f"Reproducibility report: {output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"Logical reproducibility check passed for {report['sample_count']} samples."
    )


if __name__ == "__main__":
    main()
