from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    label: int | None
    label_text: str
    captured_at: float
    completed_at: float
    latency_ms: float
    error: str | None = None

    @property
    def succeeded(self):
        return self.error is None and self.label is not None
