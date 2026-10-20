"""Compatibility imports and training CLI: python -m src.train."""
import argparse
import csv
import sys
from pathlib import Path

from src.data import DatasetCatalog
from src.data.api import count_summary, load_dataset
from src.data.constants import CLASS_NAMES, SPLIT_NAMES, REQUIRED_IMAGE_COLUMNS, REQUIRED_LOCATION_COLUMNS
from src.exceptions import TrainingError
from src.features import FeatureConfig, ImageFeatureExtractor, extract_features, feature_block_sizes
from src.preprocessing import PreprocessingConfig, load_and_preprocess
from src.storage import ArtifactStore
from src.training.api import run_training
from src.training.classifier_factory import make_classifier
from src.training.constants import CLASSIFIERS, DEFAULT_GRIDS, FEATURE_SETS, SELECTION_RULE
from src.training.feature_matrix import extract_matrix, feature_columns
from src.training.metrics import score_predictions
from src.training.model_trainer import ModelTrainer
from src.training.search import expand_grid, selection_key
from src.training.training_config import TrainingConfig
from src.utils.file_io import read_table, resolve_image_path, sha256_file, write_atomic
from src.utils.serialization import to_python


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root / "dataset")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS), default=list(FEATURE_SETS))
    parser.add_argument("--classifiers", nargs="+", choices=list(CLASSIFIERS), default=list(CLASSIFIERS))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=("squash", "center_crop", "pad"), default="squash")
    parser.add_argument("--pad-value", type=int, default=0)
    parser.add_argument("--hog-cell", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_unverified and (args.models_dir is None or args.results_dir is None):
        print("ERROR: --allow-unverified requires explicit --models-dir and --results-dir.", file=sys.stderr)
        return 1
    try:
        result = run_training(
            args.dataset_dir, args.models_dir or root / "models", args.results_dir or root / "results",
            tuple(args.feature_sets), tuple(args.classifiers), args.image_size,
            args.resize_mode, args.pad_value, args.hog_cell, args.seed, args.n_jobs,
            not args.no_class_weight, args.allow_unverified,
        )
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    report = result["report"]
    metrics = report["validation_metrics"]
    print("\nSelected:", report["feature_set"], "+", report["classifier"], report["classifier_params"])
    print(
        f"Validation macro-F1={metrics['macro_f1']:.3f}, "
        f"balanced accuracy={metrics['balanced_accuracy']:.3f}, accuracy={metrics['accuracy']:.3f}"
    )
    print(f"Saved {result['models_dir'] / 'pipeline.joblib'} and results in {result['results_dir']}")
    print("The test split was not opened. Evaluate it only after the configuration is frozen.")
    if report["development_run"]:
        print("DEVELOPMENT RUN: do not report these results as final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
