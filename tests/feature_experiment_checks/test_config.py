import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from tests.feature_experiment_checks.helpers import fixture
from src.experiments.feature_ablation.feature_experiment_config import FeatureExperimentConfig


class ConfigTests(unittest.TestCase):
    def test_root_relative_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "config.json"
            path.write_text(json.dumps({
                "dataset_dir": "dataset", "reference_model": "reference/pipeline.joblib",
                "reference_metadata": "reference/meta.json", "reference_result": "reference/result.json", "plan_dir": "plan",
            }))
            config = FeatureExperimentConfig.from_json(path, root)
            self.assertEqual(config.dataset_dir, (root / "dataset").resolve())

    def test_configuration_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            with self.assertRaises(FrozenInstanceError):
                config.experiment_id = "changed"

    def test_unknown_json_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"bad": 1}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                FeatureExperimentConfig.from_json(path, folder)

    def test_invalid_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            for change in ({"allow_unverified": "yes"}, {"experiment_id": "other"}, {"camera_diagnostic_status": "guessed"}, {"plan_dir": config.dataset_dir / "plan"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    replace(config, **change)
