import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.application.application import Application
from src.camera.camera_application import CameraApplication
from src.camera.camera_config import CameraConfig


class CameraApplicationTests(unittest.TestCase):
    def backend(self):
        cv = Mock()
        cv.waitKey.return_value = ord("q")
        cv.FONT_HERSHEY_SIMPLEX = 0
        cv.LINE_AA = 16
        return cv

    def test_unified_entry_point_routes_camera_without_training(self):
        app = Application()
        with patch.object(app, "run_camera", return_value=0) as camera, patch.object(app, "run_train") as train:
            self.assertEqual(app.run(["camera", "--model", "sample.joblib"]), 0)
            camera.assert_called_once_with(["--model", "sample.joblib"])
            train.assert_not_called()

    def test_existing_all_command_still_stops_if_tests_fail(self):
        app = Application()
        with patch.object(app, "run_tests", return_value=1), patch.object(app, "run_train") as train, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(app.run(["all"]), 1)
            train.assert_not_called()

    def test_preview_quit_releases_camera_window_and_worker(self):
        cv = self.backend()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch("src.camera.camera_application.ModelPredictor"), patch("src.camera.camera_application.PredictionWorker") as worker_cls, patch("src.camera.camera_application.CameraSource") as source_cls:
            source_cls.return_value.read.return_value = frame
            worker_cls.return_value.latest_result.return_value = None
            worker_cls.return_value.close.return_value = True
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(CameraApplication(CameraConfig("model.joblib"), cv_module=cv).run(), 0)
            source_cls.return_value.close.assert_called_once()
            worker_cls.return_value.close.assert_called_once()
            cv.destroyAllWindows.assert_called_once()
            self.assertFalse(np.any(frame))

    def test_worker_cleanup_runs_even_when_window_cleanup_fails(self):
        cv = self.backend()
        cv.destroyAllWindows.side_effect = RuntimeError("window cleanup")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch("src.camera.camera_application.ModelPredictor"), patch("src.camera.camera_application.PredictionWorker") as worker_cls, patch("src.camera.camera_application.CameraSource") as source_cls:
            source_cls.return_value.read.return_value = frame
            worker_cls.return_value.latest_result.return_value = None
            worker_cls.return_value.close.return_value = True
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                CameraApplication(CameraConfig("model.joblib"), cv_module=cv).run()
            source_cls.return_value.close.assert_called_once()
            worker_cls.return_value.close.assert_called_once()
