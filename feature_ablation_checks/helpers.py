import csv
import json
from inspect import signature
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.feature_ablation.experiment_plan_store import ExperimentPlanStore
from src.experiments.feature_ablation.feature_experiment_config import FeatureExperimentConfig
from src.experiments.feature_ablation.feature_experiment_planner import (
    FeatureExperimentPlanner, json_safe, sha256_file,
)
from src.features import FeatureConfig, ImageFeatureExtractor
from src.preprocessing import PreprocessingConfig


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def logistic_regression():
    values = {"C": 0.01, "solver": "liblinear", "max_iter": 10000, "tol": 1e-4,
              "random_state": 42, "class_weight": "balanced"}
    parameter = signature(LogisticRegression).parameters.get("penalty")
    if parameter is None or parameter.default == "deprecated":
        values["l1_ratio"] = 0.0
    else:
        values["penalty"] = "l2"
    return LogisticRegression(**values)


def build_case(root, perturb_reference=False, camera_status="pending"):
    root = Path(root)
    dataset = root / "dataset"
    dataset.mkdir()
    images, locations = [], []
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for label in (0, 1):
            group = f"{split}_{label}"
            locations.append({"group_id": group, "class_name": "inside" if label else "outside", "boundary_verified": "yes"})
            for index in range(count):
                name = f"{group}_{index}.png"
                if split == "test":
                    (dataset / name).write_bytes(b"test image must never be decoded")
                else:
                    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
                    stripe = (np.arange(64) // 8 % 2) * 170
                    pixels[:] = stripe[None, :, None] if label else stripe[:, None, None]
                    pixels[:, :, label] = np.minimum(pixels[:, :, label].astype(int) + 40 + index * 3, 255)
                    Image.fromarray(pixels).save(dataset / name)
                images.append({"image_id": name, "relative_path": name, "label": str(label),
                               "group_id": group, "split": split})
    write_csv(dataset / "images.csv", images)
    write_csv(dataset / "locations.csv", locations)
    by_split = {name: [dict(row) for row in images if row["split"] == name] for name in ("train", "validation", "test")}
    paths = {name: [dataset / row["relative_path"] for row in by_split[name]] for name in ("train", "validation")}
    labels = {name: [int(row["label"]) for row in by_split[name]] for name in paths}
    features = ImageFeatureExtractor(
        PreprocessingConfig(64),
        FeatureConfig(use_color=False, use_hog=True, use_lbp=False, hog_pixels_per_cell=16))
    pipeline = Pipeline([("features", features), ("scale", StandardScaler()), ("classifier", logistic_regression())])
    pipeline.fit(paths["train"], labels["train"])
    predictions = pipeline.predict(paths["validation"])
    metrics = {
        "accuracy": float(accuracy_score(labels["validation"], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels["validation"], predictions)),
        "macro_f1": float(f1_score(labels["validation"], predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels["validation"], predictions, labels=[0, 1]).tolist(),
    }
    if perturb_reference:
        metrics["accuracy"] = max(0.0, metrics["accuracy"] - 0.25)
    candidate = {"id": "hog_lr_l2_c_0_01", "family": "logistic_regression", "penalty": "l2", "C": 0.01}
    reference_dir = root / "reference/best"
    reference_dir.mkdir(parents=True)
    joblib.dump(pipeline, reference_dir / "pipeline.joblib")
    metadata = {
        "candidate": candidate, "feature_set": "hog",
        "feature_config": features.feature_config.to_dict(),
        "preprocessing_config": features.preprocessing_config.to_dict(),
        "estimator_params": json_safe(pipeline.named_steps["classifier"].get_params(deep=False)),
        "images_csv_sha256": sha256_file(dataset / "images.csv"),
    }
    result = {"candidate": candidate, "validation_metrics": metrics}
    (reference_dir / "pipeline_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "reference/best_result.json").write_text(json.dumps(result), encoding="utf-8")
    config = FeatureExperimentConfig(
        dataset, reference_dir / "pipeline.joblib", reference_dir / "pipeline_metadata.json",
        root / "reference/best_result.json", root / "plan", camera_diagnostic_status=camera_status)
    loader = lambda settings: {"dataset_dir": dataset, "rows": images, "by_split": by_split}
    prepared = FeatureExperimentPlanner(config, dataset_loader=loader).prepare()
    ExperimentPlanStore(config.plan_dir).freeze(prepared)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    return {"root": root, "config": config, "config_path": config_path, "dataset": dataset,
            "by_split": by_split, "paths": paths, "predictions": predictions}


def tree_digest(directory):
    return {str(path.relative_to(directory)): sha256_file(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}
