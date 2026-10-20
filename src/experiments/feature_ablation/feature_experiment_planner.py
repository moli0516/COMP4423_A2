import csv
import hashlib
import io
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .feature_pipeline_factory import FeaturePipelineFactory
from .feature_variant import FeatureVariant


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureExperimentPlanner:
    """Validate the fixed protocol and construct four UNFITTED pipelines only."""

    def __init__(self, config, *, dataset_loader=None, dimension_function=None):
        self.config = config
        self.dataset_loader = dataset_loader or self._load_dataset
        self.dimension_function = dimension_function

    @staticmethod
    def _load_dataset(config):
        from src.data.dataset_catalog import DatasetCatalog
        return DatasetCatalog(config.dataset_dir, config.allow_unverified).load()

    def prepare(self):
        config = self.config
        source_paths = {
            "images_csv": config.dataset_dir / "images.csv",
            "locations_csv": config.dataset_dir / "locations.csv",
            "reference_model": config.reference_model,
            "reference_metadata": config.reference_metadata,
            "reference_result": config.reference_result,
        }
        hashes = {name: sha256_file(path) for name, path in source_paths.items()}
        metadata = json.loads(config.reference_metadata.read_text(encoding="utf-8"))
        result = json.loads(config.reference_result.read_text(encoding="utf-8"))
        if metadata.get("images_csv_sha256") != hashes["images_csv"]:
            raise ValueError("Current images.csv differs from the retained model; do not resplit before this experiment.")
        candidate = metadata.get("candidate", {})
        if candidate.get("family") != "logistic_regression" or candidate.get("penalty") != "l2" or candidate.get("C") != 0.01:
            raise ValueError("Reference metadata must identify the retained L2 LR C=0.01 candidate.")
        if result.get("candidate") != candidate:
            raise ValueError("Reference best_result and metadata identify different candidates.")
        pipeline = joblib.load(config.reference_model)
        factory = FeaturePipelineFactory(pipeline, self.dimension_function)
        reference_features = pipeline.named_steps["features"]
        if metadata.get("feature_config") != reference_features.feature_config.to_dict():
            raise ValueError("Reference model and metadata disagree on feature configuration.")
        if metadata.get("preprocessing_config") != reference_features.preprocessing_config.to_dict():
            raise ValueError("Reference model and metadata disagree on preprocessing.")
        if metadata.get("estimator_params") != json_safe(pipeline.named_steps["classifier"].get_params(deep=False)):
            raise ValueError("Reference model and metadata disagree on estimator parameters.")
        data = self.dataset_loader(config)
        subsets = data["by_split"]
        group_sets = {}
        counts = {}
        for name in ("train", "validation", "test"):
            rows = subsets.get(name, [])
            if {str(row["label"]) for row in rows} != {"0", "1"}:
                raise ValueError(f"Both classes are required in {name}.")
            group_sets[name] = {row["group_id"] for row in rows}
            counts[name] = {
                label_name: {
                    "images": sum(str(row["label"]) == label for row in rows),
                    "groups": len({row["group_id"] for row in rows if str(row["label"]) == label}),
                }
                for label_name, label in (("outside", "0"), ("inside", "1"))
            }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"Location group leakage: {left}/{right}")
        image_hashes = {}
        for name in ("train", "validation"):
            for row in subsets[name]:
                path = (config.dataset_dir / row["relative_path"]).resolve()
                if not path.is_relative_to(config.dataset_dir):
                    raise ValueError("Image path escapes the dataset directory.")
                image_hashes[path.relative_to(config.dataset_dir).as_posix()] = sha256_file(path)
        descriptions = [factory.describe(variant) for variant in FeatureVariant.planned_variants()]
        estimator_parameters = descriptions[0]["estimator_parameters"]
        if any(record["estimator_parameters"] != estimator_parameters for record in descriptions):
            raise ValueError("Estimator settings must be identical for all feature variants.")
        snapshots = {name: path.read_bytes() for name, path in source_paths.items() if name in {"images_csv", "locations_csv"}}
        if any(sha256_file(path) != hashes[name] for name, path in source_paths.items()):
            raise ValueError("Reference files or CSV changed during planning.")
        plan = json_safe({
            "experiment_id": config.experiment_id,
            "stage": "prepared_architecture_only",
            "training_implemented": False,
            "completed_fits": 0,
            "planned_fits": 4,
            "question": "With fixed L2 LR C=0.01 and the existing grouped split, does adding global HSV, spatial HSV or LBP improve development macro-F1 over HOG alone?",
            "config": config.to_dict(),
            "variants": descriptions,
            "counts": counts,
            "reference_candidate": candidate,
            "reference_validation_metrics": result["validation_metrics"],
            "input_files": {name: {"path": str(path), "sha256": hashes[name]} for name, path in source_paths.items()},
            "train_validation_image_hashes": image_hashes,
            "dataset_records": [dict(row) for rows in subsets.values() for row in rows],
            "rules": {
                "selection": ["macro_f1", "balanced_accuracy", "accuracy", "earlier_variant_on_exact_tie"],
                "secondary_metrics": ["outside_recall", "inside_recall", "inside_precision", "confusion_matrix"],
                "eligibility": "finite outputs, no ConvergenceWarning or iteration-limit event",
                "fit_on": ["train"], "select_on": ["validation"],
                "resplit": False, "threshold_search": False, "C_search": False,
                "supervised_feature_selection": False, "test_evaluation": False,
                "stop_after": "four planned variants; preserve negative findings",
            },
            "camera_diagnostic_status": config.camera_diagnostic_status,
            "training_gate": "Training is not implemented in this architecture package. Resolve camera inference consistency before adding or running the training stage.",
            "test_images_hashed_or_decoded": 0,
            "warnings": [
                "Development-only comparison; labels/boundaries still require verification.",
                "The reference validation score was already used for model selection.",
                "camera_diagnostic_status is a user record, not an automated camera check.",
            ],
            "source_sha256": {path.name: sha256_file(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))},
            "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        })
        return {"plan": plan, "factory": factory, "snapshots": snapshots}
