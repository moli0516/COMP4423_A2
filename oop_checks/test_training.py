import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from oop_checks.helpers import make_dataset
from src.preprocessing import ImagePreprocessor
from src.train import run_training, score_predictions, to_python
from src.training.constants import FEATURE_SETS
from src.training.search import expand_grid, selection_key
from src.utils.file_io import write_atomic


class TrainingTests(unittest.TestCase):
    def test_numpy_json_conversion(self):
        result = to_python({"a": np.array([[1, 2]]), "b": (np.int64(3), np.float32(1.5))})
        self.assertEqual(result, {"a": [[1, 2]], "b": [3, 1.5]})
        json.dumps(result)

    def test_metrics_confusion_matrix_is_list(self):
        scores = score_predictions([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(scores["confusion_matrix"], [[1, 1], [0, 2]])
        json.dumps(scores)

    def test_search_and_selection(self):
        self.assertEqual(len(list(expand_grid([{"C": [1, 10], "kernel": ["linear"]}]))), 2)
        self.assertEqual(selection_key({"macro_f1": 0.5, "balanced_accuracy": 0.6, "accuracy": 0.7}), (0.5, 0.6, 0.7))

    def test_atomic_writer_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text("old")
            def fail(handle):
                handle.write("partial")
                raise ValueError("expected")
            with self.assertRaises(ValueError):
                write_atomic(path, fail)
            self.assertEqual(path.read_text(), "old")
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_full_40_trial_training_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images, _ = make_dataset(root / "dataset")
            opened = []
            original_load = ImagePreprocessor.load
            def guarded_load(instance, path):
                if Path(path).name.startswith("test_"):
                    raise AssertionError("Test image was opened")
                opened.append(str(path))
                return original_load(instance, path)
            with patch.object(ImagePreprocessor, "load", guarded_load):
                result = run_training(
                    root / "dataset", root / "models", root / "results",
                    image_size=64, hog_cell=16, log=lambda message: None,
                )
            self.assertEqual(len(result["trials"]), 40)
            self.assertEqual(result["report"]["test_images_opened_by_training"], 0)
            self.assertTrue(opened)
            for model_file in ("pipeline.joblib", "pipeline_metadata.json"):
                self.assertTrue((root / "models" / model_file).is_file())
            for result_file in ("validation_experiments.csv", "training_report.json"):
                self.assertTrue((root / "results" / result_file).is_file())
            json.loads((root / "models/pipeline_metadata.json").read_text())
            json.loads((root / "results/training_report.json").read_text())
            loaded = joblib.load(root / "models/pipeline.joblib")
            paths = [root / "dataset" / row["relative_path"] for row in images if row["split"] == "validation"]
            np.testing.assert_array_equal(loaded.predict(paths), result["pipeline"].predict(paths))
            self.assertEqual(result["report"]["top_trials"][0], result["report"]["best_trial"])
            self.assertEqual(set(FEATURE_SETS), {trial["feature_set"] for trial in result["trials"]})
