import importlib
import time

from .camera_source import CameraSource
from .model_predictor import ModelPredictor
from .prediction_worker import PredictionWorker


class CameraApplication:
    """Continuous preview with periodically sampled, asynchronous model results."""

    WINDOW = "PolyU campus camera - Q or Esc to quit"

    def __init__(self, config, *, cv_module=None):
        self.config = config
        self.cv2 = cv_module if cv_module is not None else importlib.import_module("cv2")

    def run(self):
        predictor = ModelPredictor(self.config.model_path)
        worker = PredictionWorker(predictor, self.config.interval_seconds)
        source = CameraSource(self.config, cv_module=self.cv2)
        last_logged = None
        window_open = False
        try:
            source.open()
            worker.start()
            self.cv2.namedWindow(self.WINDOW, self.cv2.WINDOW_NORMAL)
            window_open = True
            print(f"Camera active. Prediction target: one every {self.config.interval_seconds:g}s.")
            print("Q / Esc: quit. Prediction refers to the last sampled frame, not every preview frame.")
            print("Development demo only; no frames or recordings are saved.")
            while True:
                frame = source.read()
                captured_at = time.monotonic()
                worker.try_submit(frame, captured_at)
                result = worker.latest_result()
                preview = self.cv2.flip(frame, 1) if self.config.mirror_preview else frame.copy()
                self._draw_status(preview, result, captured_at)
                self.cv2.imshow(self.WINDOW, preview)
                if result is not None and result is not last_logged:
                    if result.succeeded:
                        print(f"{result.label_text}; inference={result.latency_ms:.1f} ms", flush=True)
                    else:
                        print(result.error, flush=True)
                    last_logged = result
                key = self.cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if self.cv2.getWindowProperty(self.WINDOW, self.cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            try:
                source.close()
            finally:
                try:
                    if window_open:
                        self.cv2.destroyAllWindows()
                finally:
                    if not worker.close():
                        print("Inference is still finishing; the daemon worker will not block process exit.")
        return 0

    def _draw_status(self, frame, result, now):
        if result is None:
            lines = ["Waiting for first prediction..."]
            color = (0, 220, 255)
        elif result.succeeded:
            lines = [
                "Last sample: " + result.label_text,
                f"Sample age: {max(0.0, now - result.captured_at):.1f}s | inference: {result.latency_ms:.1f}ms",
            ]
            color = (0, 220, 0) if result.label == 1 else (0, 180, 255)
        else:
            lines = ["Prediction error - see terminal", result.error[:90]]
            color = (0, 0, 255)
        lines.append(f"Target: every {self.config.interval_seconds:g}s | Q/Esc: quit")
        lines.append("DEVELOPMENT DEMO - no location verification")
        for index, text in enumerate(lines):
            position = (12, 28 + index * 26)
            self.cv2.putText(frame, text, position, self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, self.cv2.LINE_AA)
            self.cv2.putText(frame, text, position, self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, self.cv2.LINE_AA)
