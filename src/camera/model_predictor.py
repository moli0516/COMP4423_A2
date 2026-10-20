import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline

from .prediction_result import PredictionResult

LABELS = {0: "Outside PolyU main campus", 1: "Inside PolyU main campus"}


class ModelPredictor:
    """Load one trusted full pipeline and feed it RGB PIL images, never raw BGR."""

    def __init__(self, model_path=None, *, pipeline=None):
        if pipeline is None:
            if model_path is None:
                raise ValueError("Provide a saved model path.")
            path = Path(model_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Model does not exist: {path}")
            pipeline = joblib.load(path)
        if not isinstance(pipeline, Pipeline) or "features" not in pipeline.named_steps:
            raise ValueError("Expected a full saved sklearn Pipeline with a 'features' step.")
        self.pipeline = pipeline

    def predict(self, frame_bgr, captured_at):
        if not isinstance(frame_bgr, np.ndarray):
            raise ValueError("Camera frame must be a NumPy array.")
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Camera frame must be a uint8 BGR image.")
        if min(frame_bgr.shape[:2]) < 1:
            raise ValueError("Camera frame must be nonempty.")
        start = time.perf_counter()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = Image.fromarray(rgb)
        values = np.asarray(self.pipeline.predict([image])).reshape(-1)
        if values.size != 1 or values[0] not in LABELS:
            raise ValueError("Model must return one binary label: outside=0 or inside=1.")
        label = int(values[0])
        return PredictionResult(
            label=label, label_text=LABELS[label], captured_at=captured_at,
            completed_at=time.monotonic(), latency_ms=(time.perf_counter() - start) * 1000,
        )
