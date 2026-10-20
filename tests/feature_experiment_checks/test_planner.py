import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.feature_experiment_checks.helpers import fixture, dimension
from src.experiments.feature_ablation.feature_experiment_planner import FeatureExperimentPlanner


class PlannerTests(unittest.TestCase):
    def test_prepare_performs_no_fit_or_decode_and_does_not_hash_test_images(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            before = (config.dataset_dir / "images.csv").read_bytes()
            planner = FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension)
            from src.experiments.feature_ablation import feature_experiment_planner as module
            original_hash = module.sha256_file
            def guarded_hash(path):
                self.assertFalse(Path(path).name.startswith("test_"))
                return original_hash(path)
            with patch.object(module, "sha256_file", guarded_hash):
                prepared = planner.prepare()
            self.assertFalse(config.plan_dir.exists())
            self.assertEqual((config.dataset_dir / "images.csv").read_bytes(), before)
            self.assertEqual(prepared["plan"]["completed_fits"], 0)
            self.assertFalse(prepared["plan"]["training_implemented"])
            self.assertEqual(len(prepared["plan"]["train_validation_image_hashes"]), 4)
            self.assertEqual(prepared["plan"]["camera_diagnostic_status"], "pending")

    def test_changed_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            with (config.dataset_dir / "images.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()

    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            data = loader(config)
            data["by_split"]["validation"][0]["group_id"] = data["by_split"]["train"][0]["group_id"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()

    def test_reference_candidate_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            result = json.loads(config.reference_result.read_text())
            result["candidate"]["C"] = 1.0
            config.reference_result.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different candidates"):
                FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()

    def test_metadata_estimator_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            metadata = json.loads(config.reference_metadata.read_text())
            metadata["estimator_params"]["tol"] = 0.02
            config.reference_metadata.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "estimator parameters"):
                FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()
