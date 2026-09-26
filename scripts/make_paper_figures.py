"""Generate reproducible paper figures from structured experiment results."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.parse_args()
    raise NotImplementedError("Paper figures are implemented after the result schema is frozen.")


if __name__ == "__main__":
    main()
