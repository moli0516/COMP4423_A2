import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from camera_checks.helpers import rgb_features
from src.camera.camera_config import CameraConfig
from src.camera.model_predictor import ModelPredictor


class ModelPredictorTests(unittest.TestCase):
    def pipeline(self, label=1):
        pipeline = Mock(spec=Pipeline)
        pipeline.named_steps = {"features": object()}
        pipeline.predict.return_value = np.array([label])
        return pipeline

    def test_bgr_is_converted_to_rgb_pil_without_mutating_input(self):
        pipeline = self.pipeline()
        predictor = ModelPredictor(pipeline=pipeline)
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:] = (0, 0, 255)
        before = frame.copy()
        result = predictor.predict(frame, 100.0)
        image = pipeline.predict.call_args.args[0][0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.label, 1)
        self.assertEqual(result.captured_at, 100.0)
        self.assertTrue(result.succeeded)
        np.testing.assert_array_equal(frame, before)

    def test_invalid_frame_types_and_shapes_are_rejected(self):
        predictor = ModelPredictor(pipeline=self.pipeline())
        for frame in (None, np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3, 3)), np.zeros((0, 3, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(frame, "shape", None)), self.assertRaises(ValueError):
                predictor.predict(frame, 0)

    def test_non_binary_label_is_rejected(self):
        predictor = ModelPredictor(pipeline=self.pipeline(3))
        with self.assertRaises(ValueError):
            predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8), 0)

    def test_missing_features_step_is_rejected(self):
        pipeline = self.pipeline()
        pipeline.named_steps = {"classifier": object()}
        with self.assertRaises(ValueError):
            ModelPredictor(pipeline=pipeline)

    def test_model_file_is_loaded_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.joblib"
            path.write_bytes(b"placeholder")
            pipeline = self.pipeline()
            with patch("src.camera.model_predictor.joblib.load", return_value=pipeline) as load:
                predictor = ModelPredictor(path)
                for _ in range(3):
                    predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8), 0)
                load.assert_called_once_with(path.resolve())

    def test_configuration_validation(self):
        for options in ({"interval_seconds": 0}, {"interval_seconds": float("nan")}, {"camera_index": -1}, {"width": 0}, {"backend": "invalid"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CameraConfig("model.joblib", **options)

    def test_saved_real_pipeline_loads_and_predicts_in_fresh_process(self):
        images = [Image.new("RGB", (20, 20), color) for color in ((0, 0, 255), (10, 10, 230), (255, 0, 0), (230, 10, 10))]
        pipeline = Pipeline([
            ("features", FunctionTransformer(rgb_features, validate=False)),
            ("classifier", LogisticRegression(random_state=0)),
        ]).fit(images, [0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.joblib"
            joblib.dump(pipeline, path)
            program = (
                "import numpy as np,sys; "
                "from src.camera.model_predictor import ModelPredictor; "
                "f=np.zeros((24,32,3),dtype=np.uint8); f[:]=(0,0,255); "
                "r=ModelPredictor(sys.argv[1]).predict(f,0); assert r.label==1"
            )
            completed = subprocess.run([sys.executable, "-c", program, str(path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)
