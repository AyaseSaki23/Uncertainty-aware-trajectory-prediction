"""Tests for the resumable AV2 Motion Forecasting download script."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_av2_motion.py"
SPEC = importlib.util.spec_from_file_location("download_av2_motion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadAV2MotionTest(unittest.TestCase):
    def test_parse_splits_accepts_spaces_commas_and_removes_duplicates(self) -> None:
        self.assertEqual(MODULE.parse_splits(["train,val", "train"]), ["train", "val"])

    def test_parse_splits_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            MODULE.parse_splits(["train", "sensor"])

    def test_child_environment_removes_proxy_variables(self) -> None:
        environment = {
            "PATH": "/usr/bin",
            "http_proxy": "http://proxy",
            "HTTPS_PROXY": "http://proxy",
        }
        cleaned = MODULE.without_proxy_variables(environment)
        self.assertEqual(cleaned, {"PATH": "/usr/bin"})
        self.assertIn("http_proxy", environment)

    def test_sync_command_is_resumable_and_targets_one_split(self) -> None:
        command = MODULE.build_sync_command(
            "/opt/bin/s5cmd",
            split="train",
            destination=Path("/data/raw/av2/train"),
            workers=12,
            dry_run=True,
            log_level="info",
        )
        self.assertIn("--no-sign-request", command)
        self.assertIn("--dry-run", command)
        self.assertIn("--size-only", command)
        self.assertIn("12", command)
        self.assertIn(
            "s3://argoverse/datasets/av2/motion-forecasting/train/*", command
        )
        self.assertEqual(command[-1], "/data/raw/av2/train/")


if __name__ == "__main__":
    unittest.main()
