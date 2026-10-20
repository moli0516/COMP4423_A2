import json
import tempfile
import unittest

from tests.feature_experiment_checks.helpers import fixture, dimension
from src.experiments.feature_ablation.feature_experiment_planner import FeatureExperimentPlanner
from src.experiments.feature_ablation.experiment_plan_store import ExperimentPlanStore


class StoreTests(unittest.TestCase):
    def test_freeze_writes_plan_variants_and_csv_snapshots_but_no_model(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            prepared = FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()
            output = ExperimentPlanStore(config.plan_dir).freeze(prepared)
            plan = json.loads((output / "experiment_plan.json").read_text())
            self.assertEqual(plan["stage"], "prepared_architecture_only")
            self.assertEqual(plan["completed_fits"], 0)
            self.assertEqual(len(list((output / "variants").glob("*.json"))), 4)
            self.assertEqual((output / "snapshots/images.csv").read_bytes(), (config.dataset_dir / "images.csv").read_bytes())
            self.assertEqual(list(output.rglob("*.joblib")), [])

    def test_existing_plan_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            prepared = FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()
            store = ExperimentPlanStore(config.plan_dir)
            store.freeze(prepared)
            with self.assertRaises(FileExistsError):
                store.freeze(prepared)

    def test_changed_image_after_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            prepared = FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()
            (config.dataset_dir / "train_0.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image changed"):
                ExperimentPlanStore(config.plan_dir).freeze(prepared)
            self.assertFalse(config.plan_dir.exists())

    def test_changed_reference_after_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, loader = fixture(folder)
            prepared = FeatureExperimentPlanner(config, dataset_loader=loader, dimension_function=dimension).prepare()
            config.reference_result.write_text("{}")
            with self.assertRaisesRegex(ValueError, "input file changed"):
                ExperimentPlanStore(config.plan_dir).freeze(prepared)
            self.assertFalse(config.plan_dir.exists())
