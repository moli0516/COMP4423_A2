import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from tests.polyu_lr_checks.helpers import fixture
from src.experiments.polyu_holdout_experiment import PolyuHoldoutExperiment


class ExperimentTests(unittest.TestCase):
    def setup_case(self, folder):
        config, loader, transformer = fixture(folder)
        runner = PolyuHoldoutExperiment(config, dataset_loader=loader, transformer_factory=transformer, log=lambda message: None)
        return config, loader, runner

    def test_dry_run_does_not_decode_images_fit_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            with patch("tests.polyu_lr_checks.hog_test_transformer.Image.open", side_effect=AssertionError("Image opened")):
                result = runner.run(dry_run=True)
            self.assertTrue(result["dry_run"])
            self.assertFalse(config.output_dir.exists())

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            config.output_dir.mkdir()
            with self.assertRaises(FileExistsError):
                runner.run(dry_run=True)

    def test_changed_csv_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            with (config.dataset_dir / "images.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                runner.preflight()

    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, loader, runner = self.setup_case(folder)
            data = loader(config)
            data["by_split"]["validation"][0]["group_id"] = data["by_split"]["train"][0]["group_id"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                runner.preflight()

    def test_baseline_reproduction_mismatch_stops_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            reference = json.loads(config.reference_report.read_text())
            reference["validation_metrics"]["accuracy"] = 0.1
            config.reference_report.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                runner.run()
            manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse((config.output_dir / "best").exists())

    def test_complete_seven_fits_preserve_csv_and_never_open_test(self):
        with tempfile.TemporaryDirectory() as folder:
            config, loader, runner = self.setup_case(folder)
            csv_before = (config.dataset_dir / "images.csv").read_bytes()
            from PIL import Image
            original_open = Image.open
            opened = []
            def guarded_open(path, *args, **kwargs):
                self.assertFalse(Path(path).name.startswith("test_"))
                opened.append(str(path))
                return original_open(path, *args, **kwargs)
            with patch("tests.polyu_lr_checks.hog_test_transformer.Image.open", guarded_open):
                result = runner.run()
            self.assertEqual(len(result["comparison"]), 7)
            self.assertEqual(result["manifest"]["status"], "completed")
            self.assertEqual((config.dataset_dir / "images.csv").read_bytes(), csv_before)
            self.assertTrue(opened)
            self.assertEqual(result["manifest"]["test_images_opened"], 0)
            with (config.output_dir / "comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 7)
            model_path = config.output_dir / "best/pipeline.joblib"
            model = joblib.load(model_path)
            paths = [config.dataset_dir / row["relative_path"] for row in loader(config)["by_split"]["validation"]]
            self.assertEqual(len(model.predict(paths)), 4)
            program = "import joblib,sys; m=joblib.load(sys.argv[1]); assert len(m.predict(sys.argv[2:]))==4"
            completed = subprocess.run([sys.executable, "-c", program, str(model_path), *map(str, paths)], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reference_feature_set_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            metadata = json.loads(config.reference_metadata.read_text())
            metadata["feature_set"] = "color"
            config.reference_metadata.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "HOG-only"):
                runner.preflight()

    def test_convergence_warning_marks_candidate_ineligible(self):
        import warnings
        from sklearn.linear_model import LogisticRegression
        original_fit = LogisticRegression.fit
        def warning_fit(model, X, y, *args, **kwargs):
            result = original_fit(model, X, y, *args, **kwargs)
            warnings.warn("intentional convergence diagnostic", ConvergenceWarning)
            return result
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            with patch.object(LogisticRegression, "fit", warning_fit):
                result = runner.run()
            self.assertTrue(result["manifest"]["numerical_review_required"])
            self.assertTrue(result["comparison"][0]["eligible"])
            self.assertTrue(all(not row["eligible"] for row in result["comparison"][1:]))
            self.assertEqual(result["manifest"]["best_candidate"], "baseline_hog_svm_linear_c_0_1")

    def test_iteration_limit_marks_candidate_ineligible(self):
        from sklearn.linear_model import LogisticRegression
        original_fit = LogisticRegression.fit
        def limited_fit(model, X, y, *args, **kwargs):
            result = original_fit(model, X, y, *args, **kwargs)
            model.n_iter_ = np.array([model.max_iter])
            return result
        with tempfile.TemporaryDirectory() as folder:
            _, _, runner = self.setup_case(folder)
            with patch.object(LogisticRegression, "fit", limited_fit):
                result = runner.run()
            self.assertTrue(all(not row["eligible"] for row in result["comparison"][1:]))
            self.assertEqual(result["manifest"]["best_candidate"], "baseline_hog_svm_linear_c_0_1")

    def test_fit_failure_is_not_silently_skipped(self):
        from sklearn.linear_model import LogisticRegression
        with tempfile.TemporaryDirectory() as folder:
            config, _, runner = self.setup_case(folder)
            with patch.object(LogisticRegression, "fit", side_effect=RuntimeError("intentional fit failure")):
                with self.assertRaisesRegex(RuntimeError, "intentional fit failure"):
                    runner.run()
            manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_fits"], 1)
            self.assertEqual(manifest["current_candidate"], "hog_lr_l2_c_0_01")
            self.assertFalse((config.output_dir / "best").exists())
