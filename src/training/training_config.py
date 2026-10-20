from dataclasses import dataclass
from pathlib import Path

from src.exceptions import TrainingError
from src.features import FeatureConfig
from src.preprocessing import PreprocessingConfig
from .constants import CLASSIFIERS, FEATURE_SETS


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str | Path
    models_dir: str | Path
    results_dir: str | Path
    feature_sets: tuple = tuple(FEATURE_SETS)
    classifiers: tuple = CLASSIFIERS
    image_size: int = 256
    resize_mode: str = "squash"
    pad_value: int = 0
    hog_cell: int = 32
    seed: int = 42
    n_jobs: int = 1
    balanced: bool = True
    allow_unverified: bool = False

    def validate(self):
        if not self.feature_sets or any(name not in FEATURE_SETS for name in self.feature_sets):
            raise TrainingError("Invalid or empty feature_sets.")
        if not self.classifiers or any(name not in CLASSIFIERS for name in self.classifiers):
            raise TrainingError("Invalid or empty classifiers.")
        if self.n_jobs == 0:
            raise TrainingError("n_jobs cannot be zero.")
        PreprocessingConfig(self.image_size, self.resize_mode, self.pad_value)
        FeatureConfig(hog_pixels_per_cell=self.hog_cell)
