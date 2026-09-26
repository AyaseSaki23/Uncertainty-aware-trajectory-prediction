"""Generate paper tables from structured experiment metrics."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise NotImplementedError("Paper tables are implemented after the result schema is frozen.")


if __name__ == "__main__":
    main()
