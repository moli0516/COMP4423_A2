import threading
import unittest
from unittest.mock import Mock

import numpy as np

from src.camera.prediction_result import PredictionResult
from src.camera.prediction_worker import PredictionWorker


class PredictionWorkerTests(unittest.TestCase):
    def frame(self):
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def worker(self, predictor, clock):
        worker = PredictionWorker(predictor, 1.0, clock=clock).start()
        self.addCleanup(worker.close)
        return worker

    def predictor(self):
        predictor = Mock()
        predictor.predict.side_effect = lambda frame, captured: PredictionResult(1, "Inside", captured, captured, 1)
        return predictor

    def test_accepts_first_frame_and_enforces_one_second_interval(self):
        now = [0.0]
        predictor = self.predictor()
        worker = self.worker(predictor, lambda: now[0])
        self.assertTrue(worker.try_submit(self.frame()))
        self.assertTrue(worker.wait_until_idle(2))
        now[0] = 0.999
        self.assertFalse(worker.try_submit(self.frame()))
        now[0] = 1.0
        self.assertTrue(worker.try_submit(self.frame()))
        self.assertTrue(worker.wait_until_idle(2))
        self.assertEqual(predictor.predict.call_count, 2)

    def test_busy_worker_drops_frames_without_backlog(self):
        started, release = threading.Event(), threading.Event()
        predictor = Mock()
        def slow(frame, captured):
            started.set()
            release.wait(3)
            return PredictionResult(0, "Outside", captured, captured, 1)
        predictor.predict.side_effect = slow
        now = [0.0]
        worker = self.worker(predictor, lambda: now[0])
        self.addCleanup(release.set)
        self.assertTrue(worker.try_submit(self.frame()))
        self.assertTrue(started.wait(2))
        now[0] = 4.0
        for _ in range(10):
            self.assertFalse(worker.try_submit(self.frame()))
        release.set()
        self.assertTrue(worker.wait_until_idle(2))
        self.assertEqual(predictor.predict.call_count, 1)
        self.assertTrue(worker.try_submit(self.frame()))
        self.assertTrue(worker.wait_until_idle(2))
        now[0] = 4.1
        self.assertFalse(worker.try_submit(self.frame()))

    def test_submitted_frame_is_copied(self):
        started, release = threading.Event(), threading.Event()
        predictor = Mock()
        observed = []
        def inspect_frame(frame, captured):
            started.set()
            release.wait(3)
            observed.append(int(frame[0, 0, 0]))
            return PredictionResult(1, "Inside", captured, captured, 1)
        predictor.predict.side_effect = inspect_frame
        worker = self.worker(predictor, lambda: 0)
        self.addCleanup(release.set)
        frame = self.frame()
        worker.try_submit(frame)
        self.assertTrue(started.wait(2))
        frame[:] = 255
        release.set()
        self.assertTrue(worker.wait_until_idle(2))
        self.assertEqual(observed, [0])

    def test_prediction_failure_is_reported_and_worker_can_recover(self):
        now = [0.0]
        predictor = self.predictor()
        predictor.predict.side_effect = ValueError("bad model")
        worker = self.worker(predictor, lambda: now[0])
        worker.try_submit(self.frame())
        self.assertTrue(worker.wait_until_idle(2))
        self.assertIn("bad model", worker.latest_result().error)
        self.assertFalse(worker.latest_result().succeeded)
        now[0] = 1.0
        predictor.predict.side_effect = lambda frame, captured: PredictionResult(0, "Outside", captured, captured, 1)
        worker.try_submit(self.frame())
        self.assertTrue(worker.wait_until_idle(2))
        self.assertTrue(worker.latest_result().succeeded)

    def test_closed_worker_rejects_frames(self):
        worker = self.worker(self.predictor(), lambda: 0)
        self.assertTrue(worker.close())
        self.assertFalse(worker.try_submit(self.frame()))

    def test_worker_cannot_start_twice(self):
        worker = self.worker(self.predictor(), lambda: 0)
        with self.assertRaises(RuntimeError):
            worker.start()
