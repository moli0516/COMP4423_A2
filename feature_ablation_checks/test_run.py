import csv
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from feature_ablation_checks.helpers import build_case, tree_digest
from feature_ablation_checks.test_plan_validation import make_runner
from src.experiments.feature_ablation import feature_experiment_runner as runner_module


class RunTests(unittest.TestCase):
    def test_full_run_uses_frozen_plan_and_exports_best_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            runner = make_runner(case)
            plan_before = tree_digest(case["config"].plan_dir)
            csv_before = {name: (case["dataset"] / name).read_bytes() for name in ("images.csv", "locations.csv")}
            original_open, original_hash = Image.open, runner_module.sha256_file
            def guarded_open(path, *args, **kwargs):
                self.assertFalse(Path(path).name.startswith("test_"))
                return original_open(path, *args, **kwargs)
            def guarded_hash(path):
                self.assertFalse(Path(path).name.startswith("test_"))
                return original_hash(path)
            with patch("PIL.Image.open", guarded_open), patch.object(runner_module, "sha256_file", guarded_hash):
                result = runner.run()
            output = case["root"] / "run"
            self.assertEqual(result["manifest"]["status"], "completed")
            self.assertEqual(result["manifest"]["completed_fits"], 4)
            self.assertEqual([row["variant_id"] for row in result["comparison"]], ["F0", "F1", "F2", "F3"])
            self.assertEqual([row["n_features"] for row in result["comparison"]], [324, 372, 516, 334])
            self.assertEqual(result["comparison"][0]["macro_f1_difference_vs_f0"], 0)
            self.assertEqual(result["comparison"][0]["predictions_changed_vs_f0"], 0)
            expected, best_key = None, None
            for row in result["comparison"]:
                key = (row["validation_macro_f1"], row["validation_balanced_accuracy"], row["validation_accuracy"])
                if row["eligible"] and (best_key is None or key > best_key):
                    expected, best_key = row["variant_id"], key
            self.assertEqual(result["manifest"]["best_variant"], expected)
            self.assertEqual(tree_digest(case["config"].plan_dir), plan_before)
            for name, content in csv_before.items():
                self.assertEqual((case["dataset"] / name).read_bytes(), content)
            with (output / "comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            for variant in ("F0", "F1", "F2", "F3"):
                directory = output / "candidates" / variant
                for name in ("pipeline.joblib", "pipeline_metadata.json", "metrics.json", "diagnostics.json", "validation_predictions.csv", "coefficients.npz"):
                    self.assertTrue((directory / name).is_file(), (variant, name))
            model_path = output / "best/pipeline.joblib"
            model = joblib.load(model_path)
            self.assertEqual(len(model.predict(case["paths"]["validation"])), 4)
            with Image.open(case["paths"]["validation"][0]) as image:
                self.assertEqual(int(model.predict([image.copy()])[0]), int(model.predict(case["paths"]["validation"][:1])[0]))
            program = "import joblib,sys; m=joblib.load(sys.argv[1]); assert len(m.predict(sys.argv[2:]))==4"
            completed = subprocess.run([sys.executable, "-c", program, str(model_path), *map(str, case["paths"]["validation"])], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reference_mismatch_stops_before_alternatives(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder, perturb_reference=True)
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                make_runner(case).run()
            manifest = json.loads((case["root"] / "run/run_manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["completed_fits"], manifest["current_variant"]), ("failed", 0, "F0"))
            self.assertFalse((case["root"] / "run/best").exists())

    def test_alternative_convergence_warning_is_recorded_and_ineligible(self):
        original_fit, calls = LogisticRegression.fit, [0]
        def warning_fit(model, X, y, *args, **kwargs):
            result = original_fit(model, X, y, *args, **kwargs)
            calls[0] += 1
            if calls[0] == 2:
                warnings.warn("intentional convergence warning", ConvergenceWarning)
            return result
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            with patch.object(LogisticRegression, "fit", warning_fit):
                result = make_runner(case).run()
            self.assertTrue(result["manifest"]["numerical_review_required"])
            self.assertFalse(result["comparison"][1]["eligible"])
            diagnostics = json.loads((case["root"] / "run/candidates/F1/diagnostics.json").read_text())
            self.assertTrue(diagnostics["convergence_warning"])

    def test_control_numerical_failure_aborts(self):
        original_fit = LogisticRegression.fit
        def limited_fit(model, X, y, *args, **kwargs):
            result = original_fit(model, X, y, *args, **kwargs)
            model.n_iter_ = np.array([model.max_iter])
            return result
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            with patch.object(LogisticRegression, "fit", limited_fit):
                with self.assertRaisesRegex(ValueError, "F0 failed numerical eligibility"):
                    make_runner(case).run()
            self.assertFalse((case["root"] / "run/best").exists())

    def test_fit_failure_is_recorded_and_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            with patch.object(LogisticRegression, "fit", side_effect=RuntimeError("fit failed")):
                with self.assertRaisesRegex(RuntimeError, "fit failed"):
                    make_runner(case).run()
            manifest = json.loads((case["root"] / "run/run_manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["completed_fits"]), ("failed", 0))

    def test_mutation_during_execution_blocks_best_export(self):
        original_fit, calls = LogisticRegression.fit, [0]
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            def mutate(model, X, y, *args, **kwargs):
                result = original_fit(model, X, y, *args, **kwargs)
                calls[0] += 1
                if calls[0] == 1:
                    with (case["dataset"] / "images.csv").open("a") as handle:
                        handle.write("\n")
                return result
            with patch.object(LogisticRegression, "fit", mutate):
                with self.assertRaisesRegex(ValueError, "Input changed during execution"):
                    make_runner(case).run()
            manifest = json.loads((case["root"] / "run/run_manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["completed_fits"]), ("failed", 4))
            self.assertFalse((case["root"] / "run/best").exists())
