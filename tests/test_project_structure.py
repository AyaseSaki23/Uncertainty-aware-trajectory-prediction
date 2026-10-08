"""Guard the architecture promised by the technical plan."""

from __future__ import annotations

import unittest
from pathlib import Path


class ProjectStructureTest(unittest.TestCase):
    def test_required_paths_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            "configs/data/av2_subset.yaml",
            "configs/data/av2_val_cv.yaml",
            "configs/model/gru_multimodal.yaml",
            "configs/experiment/raw_clean.yaml",
            "configs/experiment/raw_noisy.yaml",
            "configs/experiment/kf_mean.yaml",
            "configs/experiment/kf_confidence.yaml",
            "src/trajectory_prediction/data/av2_dataset.py",
            "src/trajectory_prediction/data/corruptions.py",
            "src/trajectory_prediction/evaluation/cv.py",
            "src/trajectory_prediction/geometry/coordinates.py",
            "src/trajectory_prediction/filtering/kalman_cv.py",
            "src/trajectory_prediction/filtering/features.py",
            "src/trajectory_prediction/models/constant_velocity.py",
            "src/trajectory_prediction/models/gru_multimodal.py",
            "src/trajectory_prediction/models/confidence_fusion.py",
            "src/trajectory_prediction/losses/multimodal.py",
            "src/trajectory_prediction/metrics/forecasting.py",
            "src/trajectory_prediction/metrics/robustness.py",
            "src/trajectory_prediction/reporting/tables.py",
            "src/trajectory_prediction/training/engine.py",
            "src/trajectory_prediction/visualization/bev.py",
            "scripts/preprocess_av2.py",
            "scripts/download_av2_motion.py",
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/visualize_predictions.py",
            "scripts/make_paper_tables.py",
            "scripts/make_paper_figures.py",
            "demo/app.py",
        ]
        missing = [path for path in required if not (root / path).is_file()]
        self.assertEqual(missing, [], f"Missing required project paths: {missing}")


if __name__ == "__main__":
    unittest.main()
