"""Tests for configuration loading and required-key validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trajectory_prediction.config import load_yaml, require_keys


class ConfigTest(unittest.TestCase):
    def test_load_yaml_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("experiment:\n  name: smoke\n", encoding="utf-8")
            self.assertEqual(load_yaml(path)["experiment"]["name"], "smoke")

    def test_reject_non_mapping_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("- invalid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                load_yaml(path)

    def test_require_keys_lists_missing_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "training"):
            require_keys({"experiment": {}}, "experiment", "training")


if __name__ == "__main__":
    unittest.main()
