"""Report missing source APIs without skipping or modifying regression tests."""
import importlib
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    contracts = {
        "src.preprocessing": [
            "PreprocessingError", "load_image", "normalise_orientation_and_colour",
            "DEFAULT_CONFIG", "RESIZE_MODES", "main",
        ],
        "src.preprocessing.api": ["load_and_preprocess", "preprocess_image"],
        "src.preprocessing.preprocessing_config": ["PreprocessingConfig.from_dict"],
        "src.features": ["DEFAULT_FEATURE_CONFIG", "main"],
        "src.features.feature_config": ["FeatureConfig.from_dict"],
        "src.features.image_feature_extractor": ["ImageFeatureExtractor"],
        "src.features.api": [
            "color_histogram_features", "hog_features", "lbp_features",
            "extract_feature_blocks", "extract_features", "feature_block_sizes", "feature_dimension",
        ],
        "src.training.model_trainer": ["ModelTrainer"],
        "src.training.training_config": ["TrainingConfig"],
        "src.training.api": ["run_training"],
        "src.storage.artifact_store": ["ArtifactStore", "json.dumps"],
    }
    failures = []
    for module_name, names in contracts.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            failures.append(f"{module_name}: {type(error).__name__}: {error}")
            continue
        for name in names:
            try:
                value = module
                for part in name.split("."):
                    value = getattr(value, part)
            except AttributeError:
                failures.append(f"Missing {module_name}.{name}")
    if not (root / "scripts/split_dataset.py").is_file():
        failures.append("Missing scripts/split_dataset.py (keep your existing script)")
    for message in failures:
        print(message)
    if failures:
        print(f"{len(failures)} source contract issue(s). Tests were not modified or skipped.")
        return 1
    print("Required API names are present; this is not a behavioural test pass result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
