import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression

from tests.feature_ablation_checks.helpers import build_case
from src.experiments.feature_ablation.feature_experiment_runner import FeatureExperimentRunner


def make_runner(case, name="run", confirmed=True):
    return FeatureExperimentRunner(case["config"], case["root"] / name, camera_diagnostic_confirmed=confirmed, log=lambda message: None)


class PlanValidationTests(unittest.TestCase):
    def test_dry_run_verifies_without_decoding_fitting_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            runner = make_runner(case, confirmed=False)
            with patch("PIL.Image.open", side_effect=AssertionError("decoded")), patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit")):
                result = runner.run(dry_run=True)
            self.assertTrue(result["dry_run"])
            self.assertEqual([item[1] for item in result["summary"]["variants"]], [324, 372, 516, 334])
            self.assertFalse((case["root"] / "run").exists())

    def test_pending_camera_gate_blocks_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            runner = make_runner(case, confirmed=False)
            with patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit")):
                with self.assertRaisesRegex(PermissionError, "camera_diagnostic_status is pending"):
                    runner.run()
            self.assertFalse((case["root"] / "run").exists())

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            (case["root"] / "run").mkdir()
            with self.assertRaises(FileExistsError):
                make_runner(case).run(dry_run=True)

    def test_output_cannot_overlap_plan_dataset_or_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            for path in (case["config"].plan_dir / "run", case["dataset"] / "run", case["config"].reference_model.parent / "run", case["root"] / "reference"):
                with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                    FeatureExperimentRunner(case["config"], path, camera_diagnostic_confirmed=True, log=lambda message: None).run(dry_run=True)

    def test_missing_plan_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            config = replace(case["config"], plan_dir=case["root"] / "missing_plan")
            with self.assertRaisesRegex(FileNotFoundError, "Frozen plan not found"):
                FeatureExperimentRunner(config, case["root"] / "run", log=lambda message: None).run(dry_run=True)

    def test_changed_inputs_after_freeze_are_rejected(self):
        for target in ("images.csv", "locations.csv", "reference", "metadata", "result", "image"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder:
                case = build_case(folder)
                paths = {
                    "images.csv": case["dataset"] / "images.csv", "locations.csv": case["dataset"] / "locations.csv",
                    "reference": case["config"].reference_model, "metadata": case["config"].reference_metadata,
                    "result": case["config"].reference_result, "image": case["paths"]["train"][0],
                }
                with paths[target].open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(ValueError, "changed|Input changed|image changed"):
                    make_runner(case).run(dry_run=True)

    def test_tampered_plan_fields_are_rejected(self):
        edits = {
            "stage": lambda plan: plan.__setitem__("stage", "other"),
            "records": lambda plan: plan["dataset_records"][0].__setitem__("label", "1"),
            "variant": lambda plan: plan["variants"][1].__setitem__("feature_dimension", 1),
            "config": lambda plan: plan["config"].__setitem__("allow_unverified", False),
        }
        for name, edit in edits.items():
            with self.subTest(edit=name), tempfile.TemporaryDirectory() as folder:
                case = build_case(folder)
                path = case["config"].plan_dir / "experiment_plan.json"
                plan = json.loads(path.read_text())
                edit(plan)
                path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    make_runner(case).run(dry_run=True)

    def test_current_config_difference_other_than_camera_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            config = replace(case["config"], allow_unverified=False)
            with self.assertRaisesRegex(ValueError, "differs from the frozen plan"):
                FeatureExperimentRunner(config, case["root"] / "run", log=lambda message: None).run(dry_run=True)

    def test_completed_status_in_current_config_opens_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            config = replace(case["config"], camera_diagnostic_status="completed")
            result = FeatureExperimentRunner(config, case["root"] / "run", log=lambda message: None).run(dry_run=True)
            self.assertTrue(result["summary"]["camera_gate"]["open"])
