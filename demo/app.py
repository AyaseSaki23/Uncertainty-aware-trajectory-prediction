"""Offline fixed-scenario playback demo entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.parse_args()
    raise NotImplementedError("The offline demo is implemented after model outputs are frozen.")


if __name__ == "__main__":
    main()
