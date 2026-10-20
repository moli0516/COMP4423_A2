import csv
import hashlib
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tests.feature_experiment_checks.fixture_feature_config import FixtureFeatureConfig
from tests.feature_experiment_checks.fixture_preprocessing_config import FixturePreprocessingConfig
from tests.feature_experiment_checks.no_decode_transformer import NoDecodeTransformer
from src.experiments.feature_ablation.feature_experiment_config import FeatureExperimentConfig


def dimension(feature_config, preprocessing_config):
    size = preprocessing_config.image_size
    value = 0
    if feature_config.use_hog:
        cells = size // feature_config.hog_pixels_per_cell
        block = feature_config.hog_cells_per_block
        value += (cells - block + 1) ** 2 * block ** 2 * feature_config.hog_orientations
    if feature_config.use_color:
        value += 3 * feature_config.color_bins * feature_config.color_grid ** 2
    if feature_config.use_lbp:
        value += feature_config.lbp_points + 2
    return value


def fixture(root):
    root = Path(root).resolve()
    dataset = root / "dataset"
    dataset.mkdir()
    images, locations = [], []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            group = f"{split}_{label}"
            relative = group + ".png"
            (dataset / relative).write_bytes(b"invalid image; planning must not decode")
            images.append({"image_id": group, "relative_path": relative, "label": str(label), "group_id": group, "split": split})
            locations.append({"group_id": group, "class_name": "inside" if label else "outside", "boundary_verified": "pending"})
    for name, rows in (("images.csv", images), ("locations.csv", locations)):
        with (dataset / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    features = NoDecodeTransformer(FixturePreprocessingConfig(), FixtureFeatureConfig())
    X = np.random.default_rng(42).normal(size=(8, 1764))
    y = np.array([0, 1] * 4)
    scaler = StandardScaler().fit(X)
    parameters = dict(C=0.01, solver="liblinear", max_iter=10000, tol=1e-4, random_state=42, class_weight="balanced")
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty is None or penalty.default == "deprecated":
        parameters["l1_ratio"] = 0.0
    else:
        parameters["penalty"] = "l2"
    classifier = LogisticRegression(**parameters).fit(scaler.transform(X), y)
    pipeline = Pipeline([("features", features), ("scale", scaler), ("classifier", classifier)])
    reference_dir = root / "reference"
    reference_dir.mkdir()
    model_path = reference_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    candidate = {"id": "hog_lr_l2_c_0_01", "family": "logistic_regression", "penalty": "l2", "C": 0.01}
    metadata = {
        "candidate": candidate,
        "images_csv_sha256": hashlib.sha256((dataset / "images.csv").read_bytes()).hexdigest(),
        "feature_config": features.feature_config.to_dict(),
        "preprocessing_config": features.preprocessing_config.to_dict(),
        "estimator_params": classifier.get_params(deep=False),
    }
    metadata_path, result_path = reference_dir / "metadata.json", reference_dir / "best_result.json"
    metadata_path.write_text(json.dumps(metadata))
    result_path.write_text(json.dumps({"candidate": candidate, "validation_metrics": {"macro_f1": 0.735}}))
    config = FeatureExperimentConfig(dataset, model_path, metadata_path, result_path, root / "prepared_plan")
    by_split = {split: [row for row in images if row["split"] == split] for split in ("train", "validation", "test")}
    loader = lambda settings: {"rows": images, "by_split": by_split}
    return config, pipeline, loader
