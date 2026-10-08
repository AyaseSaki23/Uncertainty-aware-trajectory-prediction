"""Evaluate a Constant Velocity baseline on a fixed AV2 validation manifest."""

from __future__ import annotations

import argparse
import sys

from trajectory_prediction.evaluation.cv import run_cv_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cv",), default="cv")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-root",
        help="AV2 root containing val/, used only if manifest cache shards are unavailable.",
    )
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    args = parser.parse_args()

    summary, paths = run_cv_evaluation(
        args.manifest,
        args.output_dir,
        data_root=args.data_root,
        miss_threshold_m=args.miss_threshold_m,
    )
    print(
        f"Evaluated {summary['sample_count']} AV2 val scenes with model={args.model}; "
        f"failed={summary['failed_scene_count']}."
    )
    print(
        f"ADE={summary['ADE']:.6f} m, FDE={summary['FDE']:.6f} m, "
        f"MissRate={summary['MissRate']:.6f}."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"Python executable: {sys.executable}")


if __name__ == "__main__":
    main()
