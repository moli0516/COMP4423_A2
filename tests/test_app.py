"""Unit tests for src.app. Run from the project root:
python -m unittest discover -s tests -t . -v
"""
import hashlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import app

WEB_DIR = Path(app.__file__).resolve().parents[1] / "webui"


class MeanColourExtractor(BaseEstimator, TransformerMixin):
    """Tiny stand-in for ImageFeatureExtractor: raw bytes in, mean RGB out."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for item in X:
            try:
                image = Image.open(io.BytesIO(item)).convert("RGB").resize((8, 8))
            except OSError as error:
                raise ValueError(str(error)) from error
            rows.append(np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0))
        return np.array(rows, dtype=np.float32).reshape(len(rows), 3)


def png_bytes(colour):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), colour).save(buffer, format="PNG")
    return buffer.getvalue()


RED, BLUE = png_bytes((220, 40, 40)), png_bytes((40, 40, 220))


def make_pipeline(classifier):
    samples = [RED, RED, BLUE, BLUE]
    return Pipeline([("features", MeanColourExtractor()), ("classifier", classifier)]).fit(samples, [1, 1, 0, 0])


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.outputs = root / "outputs"
        for folder, classifier in (
            ("run_01/best", LogisticRegression()),
            ("run_01/candidates/forest", RandomForestClassifier(n_estimators=5, random_state=0)),
        ):
            target = self.outputs / folder
            target.mkdir(parents=True)
            joblib.dump(make_pipeline(classifier), target / "pipeline.joblib")
        images_csv = "image_id,relative_path,label,group_id,split\n" + "\n".join([
            "a,x/a.png,1,in_001,train", "b,x/b.png,1,in_001,train", "c,x/c.png,1,in_002,test",
            "d,x/d.png,0,out_001,train", "e,x/e.png,0,out_002,test"]) + "\n"
        (root / "images.csv").write_text(images_csv, encoding="utf-8")
        (root / "locations.csv").write_text(
            "group_id,class_name,location_description,boundary_verified\n"
            "in_001,inside,Library,yes\nin_002,inside,Core M,yes\nout_001,outside,Sheung Shui,yes\nout_002,outside,Mira Place,yes\n",
            encoding="utf-8")
        (self.outputs / "run_01/best/pipeline_metadata.json").write_text(
            json.dumps({"trained_on_splits": ["train"], "images_csv_sha256": hashlib.sha256(images_csv.encode()).hexdigest()}),
            encoding="utf-8")
        self.dataset = root
        self.server = app.create_server("127.0.0.1", 0, self.outputs, self.dataset, WEB_DIR)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, path, data=None, content_type="application/octet-stream"):
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def json(self, path, data=None, content_type="application/octet-stream"):
        status, body = self.request(path, data, content_type)
        return status, json.loads(body.decode("utf-8"))


class PredictionTests(AppTestCase):
    def test_labels_text_and_score_for_both_classes(self):
        status, inside = self.json("/api/predict", RED)
        self.assertEqual((status, inside["label"], inside["text"]), (200, 1, "Inside PolyU main campus"))
        self.assertGreater(inside["score"], 0)
        status, outside = self.json("/api/predict", BLUE)
        self.assertEqual((status, outside["label"], outside["text"]), (200, 0, "Outside PolyU main campus"))
        self.assertLess(outside["score"], 0)

    def test_same_bytes_give_same_answer(self):
        first = self.json("/api/predict", RED)[1]
        second = self.json("/api/predict", RED)[1]
        self.assertEqual((first["label"], first["score"]), (second["label"], second["score"]))

    def test_unreadable_and_empty_bodies_are_rejected(self):
        status, body = self.json("/api/predict", b"not an image")
        self.assertEqual(status, 422)
        self.assertIn("Could not read", body["error"])
        self.assertEqual(self.json("/api/predict", b"")[0], 400)

    def test_oversized_upload_is_rejected(self):
        with mock.patch.object(app, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual(self.json("/api/predict", RED)[0], 413)


class ModelTests(AppTestCase):
    def test_best_model_is_listed_first_and_active(self):
        status, data = self.json("/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["active"], "run_01/best")
        self.assertEqual([m["id"] for m in data["models"]], ["run_01/best", "run_01/candidates/forest"])
        self.assertTrue(data["models"][0]["is_best"])

    def test_selecting_another_model_changes_predictions_source(self):
        body = json.dumps({"id": "run_01/candidates/forest"}).encode()
        self.assertEqual(self.json("/api/models/select", body, "application/json")[0], 200)
        result = self.json("/api/predict", RED)[1]
        self.assertEqual(result["model"], "run_01/candidates/forest")
        self.assertEqual(result["label"], 1)
        self.assertIn("probability", result["score_kind"])
        self.assertEqual(self.json("/api/models/active")[1]["active"], "run_01/candidates/forest")

    def test_unknown_or_path_like_ids_are_rejected(self):
        for value in ("../../etc/passwd", "nope", None, 5):
            body = json.dumps({"id": value}).encode()
            self.assertEqual(self.json("/api/models/select", body, "application/json")[0], 400, value)
        self.assertEqual(self.json("/api/models")[1]["active"], "run_01/best")

    def test_choice_is_remembered_after_restart(self):
        body = json.dumps({"id": "run_01/candidates/forest"}).encode()
        self.json("/api/models/select", body, "application/json")
        self.assertEqual(app.ModelStore(self.outputs).active_id, "run_01/candidates/forest")

    def test_pipeline_description_lists_steps(self):
        data = self.json("/api/models/active")[1]
        self.assertEqual([s["name"] for s in data["pipeline"]["steps"]], ["features", "classifier"])
        self.assertEqual(data["metadata"]["trained_on_splits"], ["train"])

    def test_no_models_reports_service_unavailable(self):
        empty = app.create_server("127.0.0.1", 0, Path(self.temp.name) / "empty", self.dataset, WEB_DIR)
        threading.Thread(target=empty.serve_forever, daemon=True).start()
        self.addCleanup(empty.server_close)
        self.addCleanup(empty.shutdown)
        req = urllib.request.Request(f"http://127.0.0.1:{empty.server_address[1]}/api/predict", data=RED, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(caught.exception.code, 503)


class DatasetAndPageTests(AppTestCase):
    def test_dataset_counts_and_hash_match(self):
        data = self.json("/api/dataset")[1]
        by_split = {row["split"]: row for row in data["summary"]}
        self.assertEqual([row["split"] for row in data["summary"]], ["train", "test"])
        self.assertEqual(by_split["train"]["inside"], {"images": 2, "groups": 1})
        self.assertEqual(by_split["train"]["outside"], {"images": 1, "groups": 1})
        self.assertEqual(by_split["test"]["total_images"], 2)
        self.assertEqual((data["total_images"], data["total_groups"]), (5, 4))
        self.assertTrue(data["matches_active_model"])

    def test_changed_images_csv_is_flagged(self):
        with (self.dataset / "images.csv").open("a", encoding="utf-8") as handle:
            handle.write("f,x/f.png,0,out_002,test\n")
        self.assertFalse(self.json("/api/dataset")[1]["matches_active_model"])

    def test_missing_dataset_is_reported_not_crashed(self):
        (self.dataset / "images.csv").unlink()
        data = self.json("/api/dataset")[1]
        self.assertFalse(data["available"])

    def test_pages_and_static_files_are_served_but_not_outside_webui(self):
        for path in ("/", "/config", "/static/app.js", "/static/style.css"):
            self.assertEqual(self.request(path)[0], 200, path)
        self.assertEqual(self.request("/static/../src/app.py")[0], 404)
        self.assertEqual(self.request("/nothing")[0], 404)


if __name__ == "__main__":
    unittest.main()
