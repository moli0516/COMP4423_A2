from .constants import CLASSIFIERS, FEATURE_SETS
from .model_trainer import ModelTrainer
from .training_config import TrainingConfig


def run_training(
    dataset_dir, models_dir, results_dir,
    feature_sets=tuple(FEATURE_SETS), classifiers=CLASSIFIERS,
    image_size=256, resize_mode="squash", pad_value=0, hog_cell=32,
    seed=42, n_jobs=1, balanced=True, allow_unverified=False,
    grids=None, log=print,
):
    config = TrainingConfig(
        dataset_dir, models_dir, results_dir, tuple(feature_sets), tuple(classifiers),
        image_size, resize_mode, pad_value, hog_cell, seed, n_jobs, balanced, allow_unverified,
    )
    return ModelTrainer(config, grids=grids, log=log).run()
