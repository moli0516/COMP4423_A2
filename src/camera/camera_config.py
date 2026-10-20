import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    model_path: str | Path
    camera_index: int = 0
    interval_seconds: float = 1.0
    width: int = 960
    height: int = 540
    backend: str = "auto"
    mirror_preview: bool = False

    def __post_init__(self):
        if type(self.camera_index) is not int or self.camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer.")
        if isinstance(self.interval_seconds, bool) or not isinstance(self.interval_seconds, (int, float)):
            raise ValueError("interval_seconds must be a positive finite number.")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive finite number.")
        for name in ("width", "height"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.backend not in {"auto", "any", "avfoundation"}:
            raise ValueError("backend must be auto, any, or avfoundation.")
