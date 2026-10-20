import importlib
import sys


class CameraSource:
    """Own camera lifetime; capture and GUI work remain on the main thread."""

    def __init__(self, config, *, cv_module=None):
        self.config = config
        self.cv2 = cv_module if cv_module is not None else importlib.import_module("cv2")
        self.capture = None

    def open(self):
        if self.capture is not None:
            raise RuntimeError("Camera is already open.")
        if self.config.backend == "avfoundation":
            if sys.platform != "darwin":
                raise ValueError("AVFoundation is available only on macOS.")
            backend = self.cv2.CAP_AVFOUNDATION
        elif self.config.backend == "auto" and sys.platform == "darwin":
            backend = self.cv2.CAP_AVFOUNDATION
        else:
            backend = self.cv2.CAP_ANY
        self.capture = self.cv2.VideoCapture(self.config.camera_index, backend)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(
                f"Cannot open camera {self.config.camera_index}. "
                "Check camera permission, device index and other camera applications."
            )
        try:
            self.capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        except Exception:
            self.close()
            raise
        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Camera returned no frame. Check connection and camera access.")
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
