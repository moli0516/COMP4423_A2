"""Unit tests for src.web_evaluation (Evaluate page backend)."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.web_evaluation import Evaluator, UnknownImageError, compute_metrics, make_thumbnail


class FakeStore:
    def __init__(self, root, meta=None):
        self.outputs_dir = Path(root) / "outputs"
        self.outputs_dir.mkdir()
        self.active_id = "best"
        self.meta = meta or {}
        self.calls = 0

    def metadata(self, model_id):
        return self.meta

    def predict(self, data, model_id=None):
        self.calls += 1
        red = Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0))[0] > 128
        return {"label": 1 if red else 0, "score": 1.5 if red else -1.5, "score_kind": "decision score"}


def make_dataset(root, flip=()):
    dataset = Path(root) / "data"
    rows = [["image_id", "relative_path", "label", "group_id", "split"]]
    for split, groups in {"validation": ["in_001", "out_001"], "test": ["in_002", "out_002"]}.items():
        for group in groups:
            inside = group.startswith("in")
            folder = dataset / "images" / ("inside" if inside else "outside") / group
            folder.mkdir(parents=True)
            for index in range(2):
                name = f"{group}_{index}"
                colour = (255, 0, 0) if inside else (0, 0, 255)
                if name in flip:
                    colour = (0, 0, 255) if inside else (255, 0, 0)
                Image.new("RGB", (40, 30), colour).save(folder / f"{name}.png")
                rows.append([name, f"images/{'inside' if inside else 'outside'}/{group}/{name}.png", int(inside), group, split])
    with (dataset / "images.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return dataset


class MetricsTests(unittest.TestCase):
    def test_known_values(self):
        metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["per_class"]["inside"]["precision"], 2 / 3)
        self.assertEqual(metrics["per_class"]["outside"]["support"], 2)
        json.dumps(metrics)

    def test_class_never_predicted(self):
        metrics = compute_metrics([0, 0, 1, 1], [1, 1, 1, 1])
        self.assertEqual(metrics["per_class"]["outside"]["f1"], 0.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def evaluator(self, flip=(), meta=None):
        store = FakeStore(self.root, meta)
        return Evaluator(store, make_dataset(self.root, flip)), store

    def test_validation_metrics_items_and_groups(self):
        evaluator, _ = self.evaluator(flip=("in_001_0",))
        result = evaluator.evaluate("validation")
        self.assertEqual(result["metrics"]["images"], 4)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.75)
        wrong = [item for item in result["items"] if not item["correct"]]
        self.assertEqual([item["image_id"] for item in wrong], ["in_001_0"])
        self.assertEqual(wrong[0]["margin"], 1.5)
        self.assertEqual(result["groups"][0]["group_id"], "in_001")
        self.assertEqual(result["groups"][0]["accuracy"], 0.5)
        json.dumps(result)

    def test_results_are_cached_until_refresh(self):
        evaluator, store = self.evaluator()
        evaluator.evaluate("validation")
        calls = store.calls
        self.assertTrue(evaluator.evaluate("validation")["cached"])
        self.assertEqual(store.calls, calls)
        self.assertFalse(evaluator.evaluate("validation", refresh=True)["cached"])

    def test_test_split_needs_confirmation_and_is_logged(self):
        evaluator, store = self.evaluator()
        with self.assertRaises(PermissionError):
            evaluator.evaluate("test")
        self.assertEqual(store.calls, 0)
        with self.assertRaises(PermissionError):
            evaluator.image_file("test", "in_002_0")
        first = evaluator.evaluate("test", confirm_test=True)
        self.assertEqual(len(first["test_history"]), 1)
        evaluator.evaluate("test", confirm_test=True, refresh=True)
        log = json.loads((store.outputs_dir / ".test_evaluations.json").read_text())
        self.assertEqual(len(log), 2)
        self.assertTrue(evaluator.image_file("test", "in_002_0").is_file())

    def test_warnings_for_training_and_selection_splits(self):
        evaluator, _ = self.evaluator(meta={"trained_on_splits": ["train"], "selected_using_splits": ["validation"], "images_csv_sha256": "old"})
        result = evaluator.evaluate("validation")
        self.assertTrue(any("choose this model" in text for text in result["warnings"]))
        self.assertTrue(any("images.csv has changed" in text for text in result["warnings"]))

    def test_bad_requests(self):
        evaluator, store = self.evaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate("dev")
        with self.assertRaises(ValueError):
            evaluator.evaluate("train")  # no train rows in this dataset
        store.active_id = None
        with self.assertRaises(LookupError):
            evaluator.evaluate("validation")

    def test_unreadable_image_is_reported_not_fatal(self):
        evaluator, _ = self.evaluator()
        (self.root / "data/images/inside/in_001/in_001_0.png").write_bytes(b"broken")
        result = evaluator.evaluate("validation")
        self.assertEqual(result["metrics"]["images"], 3)
        self.assertEqual(result["failures"][0]["image_id"], "in_001_0")

    def test_image_file_is_limited_to_the_split_and_thumbnail_works(self):
        evaluator, _ = self.evaluator()
        self.assertTrue(evaluator.image_file("validation", "in_001_0").is_file())
        with self.assertRaises(UnknownImageError):
            evaluator.image_file("validation", "in_002_0")
        big = self.root / "big.png"
        Image.new("RGB", (400, 200), (10, 20, 30)).save(big)
        self.assertEqual(Image.open(io.BytesIO(make_thumbnail(big, 100))).size, (100, 50))


if __name__ == "__main__":
    unittest.main()
