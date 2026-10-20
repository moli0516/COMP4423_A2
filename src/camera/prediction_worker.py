import math
import threading
import time

from .prediction_result import PredictionResult


class PredictionWorker:
    """One inference at a time, no frame backlog or catch-up bursts."""

    def __init__(self, predictor, interval_seconds=1.0, *, clock=time.monotonic):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive and finite.")
        self.predictor = predictor
        self.interval_seconds = interval_seconds
        self.clock = clock
        self._condition = threading.Condition()
        self._thread = None
        self._stopping = False
        self._busy = False
        self._pending = None
        self._latest = None
        self._next_due = float("-inf")

    def start(self):
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("Worker can be started only once.")
            self._thread = threading.Thread(target=self._run, name="camera-predictor", daemon=True)
            self._thread.start()
        return self

    def try_submit(self, frame, captured_at=None):
        with self._condition:
            now = self.clock()
            if self._thread is None or self._stopping or self._busy or now < self._next_due:
                return False
            self._pending = (frame.copy(), now if captured_at is None else captured_at)
            self._busy = True
            self._next_due = now + self.interval_seconds
            self._condition.notify()
            return True

    def latest_result(self):
        with self._condition:
            return self._latest

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                if self._stopping:
                    return
                frame, captured_at = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.predictor.predict(frame, captured_at)
            except Exception as error:
                result = PredictionResult(
                    label=None, label_text="Prediction failed", captured_at=captured_at,
                    completed_at=self.clock(), latency_ms=(time.perf_counter() - start) * 1000,
                    error=f"{type(error).__name__}: {error}",
                )
            with self._condition:
                if not self._stopping:
                    self._latest = result
                self._busy = False
                self._condition.notify_all()

    def wait_until_idle(self, timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda: not self._busy, timeout=timeout)

    def close(self, timeout=5.0):
        with self._condition:
            self._stopping = True
            if self._pending is not None:
                self._pending = None
                self._busy = False
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True
