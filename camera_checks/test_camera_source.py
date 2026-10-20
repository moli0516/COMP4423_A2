import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.camera.camera_config import CameraConfig
from src.camera.camera_source import CameraSource


class CameraSourceTests(unittest.TestCase):
    def backend(self):
        cv = Mock()
        cv.CAP_ANY = 0
        cv.CAP_AVFOUNDATION = 1200
        cv.CAP_PROP_FRAME_WIDTH = 3
        cv.CAP_PROP_FRAME_HEIGHT = 4
        return cv

    def test_camera_context_opens_reads_and_releases(self):
        cv = self.backend()
        capture = cv.VideoCapture.return_value
        capture.isOpened.return_value = True
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        capture.read.return_value = (True, frame)
        with CameraSource(CameraConfig("model.joblib", backend="any"), cv_module=cv) as source:
            self.assertIs(source.read(), frame)
        capture.release.assert_called_once()
        self.assertIsNone(source.capture)

    def test_failed_open_releases_capture(self):
        cv = self.backend()
        capture = cv.VideoCapture.return_value
        capture.isOpened.return_value = False
        with self.assertRaises(RuntimeError):
            CameraSource(CameraConfig("model.joblib"), cv_module=cv).open()
        capture.release.assert_called_once()

    def test_failed_read_is_reported(self):
        cv = self.backend()
        source = CameraSource(CameraConfig("model.joblib"), cv_module=cv)
        source.capture = Mock()
        source.capture.read.return_value = (False, None)
        with self.assertRaises(RuntimeError):
            source.read()
        source.close()

    def test_macos_auto_backend(self):
        cv = self.backend()
        cv.VideoCapture.return_value.isOpened.return_value = True
        with patch("src.camera.camera_source.sys.platform", "darwin"):
            source = CameraSource(CameraConfig("model.joblib"), cv_module=cv).open()
            cv.VideoCapture.assert_called_once_with(0, cv.CAP_AVFOUNDATION)
            source.close()

    def test_property_failure_releases_capture(self):
        cv = self.backend()
        cv.VideoCapture.return_value.isOpened.return_value = True
        cv.VideoCapture.return_value.set.side_effect = RuntimeError("property failure")
        with self.assertRaises(RuntimeError):
            CameraSource(CameraConfig("model.joblib"), cv_module=cv).open()
        cv.VideoCapture.return_value.release.assert_called_once()
