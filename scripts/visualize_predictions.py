"""Generate global/local CV prediction plots from a saved evaluation run."""

from __future__ import annotations

import argparse

from trajectory_prediction.visualization.results import generate_review_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection",
        choices=("review", "all"),
        default="review",
        help="review selects distinct straight, turn, and high-FDE cases.",
    )
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument(
        "--scenario-id",
        action="append",
        help="Render an explicit scenario ID; may be supplied more than once.",
    )
    args = parser.parse_args()

    result = generate_review_visualizations(
        args.run_dir,
        args.output_dir,
        scenario_ids=args.scenario_id,
        selection=args.selection,
        max_scenes=args.max_scenes,
    )
    print(f"Generated {result['generated_scene_count']} review figures.")
    for case in result["cases"]:
        print(
            f"{case['review_category']}: {case['scenario_id']} | "
            f"ADE={case['ADE']:.6f} m | FDE={case['FDE']:.6f} m | "
            f"heading_change={case['heading_change_rad']:.6f} rad"
        )


if __name__ == "__main__":
    main()
