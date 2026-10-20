import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import PIL
import skimage
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DatasetCatalog
from src.data.api import count_summary
from src.exceptions import TrainingError
from src.features import FeatureConfig, ImageFeatureExtractor
from src.preprocessing import PreprocessingConfig
from src.storage import ArtifactStore
from src.utils.file_io import resolve_image_path
from src.utils.serialization import to_python
from .classifier_factory import make_classifier
from .constants import DEFAULT_GRIDS, FEATURE_SETS, SELECTION_RULE
from .feature_matrix import extract_matrix, feature_columns
from .metrics import score_predictions
from .search import expand_grid, selection_key


class ModelTrainer:
    """Coordinate the workflow; dataset and storage dependencies are injectable."""

    def __init__(self, config, *, grids=None, log=print, catalog=None, artifact_store=None):
        self.config = config
        self.grids = DEFAULT_GRIDS if grids is None else grids
        self.log = log
        self.catalog = catalog if catalog is not None else DatasetCatalog(
            config.dataset_dir, config.allow_unverified
        )
        self.artifact_store = artifact_store if artifact_store is not None else ArtifactStore(
            config.models_dir, config.results_dir
        )

    def run(self):
        config = self.config
        config.validate()
        for name in config.classifiers:
            if name not in self.grids or not list(expand_grid(self.grids[name])):
                raise TrainingError(f"No parameter settings for classifier {name}.")
        data = self.catalog.load()
        preprocessing = PreprocessingConfig(config.image_size, config.resize_mode, config.pad_value)
        selected_configs = {
            name: FeatureConfig(**{"hog_pixels_per_cell": config.hog_cell, **FEATURE_SETS[name]})
            for name in config.feature_sets
        }
        all_config = FeatureConfig(
            use_color=any(item.use_color for item in selected_configs.values()),
            use_hog=any(item.use_hog for item in selected_configs.values()),
            use_lbp=any(item.use_lbp for item in selected_configs.values()),
            hog_pixels_per_cell=config.hog_cell,
        )
        self.log("Extracting features from train and validation images only...")
        x_train, y_train, x_val, y_val = self._prepare_features(data, preprocessing, all_config)
        trials, best = self._run_experiments(
            preprocessing, all_config, selected_configs, x_train, y_train, x_val, y_val
        )
        pipeline = self._build_pipeline(preprocessing, selected_configs[best["feature_set"]], best)
        self._verify_pipeline(pipeline, data, x_val, best)
        metadata, report = self._build_reports(data, preprocessing, selected_configs, trials, best)
        self.artifact_store.save(pipeline, metadata, report, trials)
        return {
            "pipeline": pipeline, "report": report, "trials": trials,
            "models_dir": Path(config.models_dir), "results_dir": Path(config.results_dir),
        }

    def _prepare_features(self, data, preprocessing, all_config):
        train = extract_matrix(data["by_split"]["train"], data["dataset_dir"], preprocessing, all_config)
        validation = extract_matrix(
            data["by_split"]["validation"], data["dataset_dir"], preprocessing, all_config
        )
        return train[0], train[1], validation[0], validation[1]

    def _run_experiments(self, preprocessing, all_config, selected_configs, x_train, y_train, x_val, y_val):
        trials, best = [], None
        for feature_set, feature_config in selected_configs.items():
            columns = feature_columns(all_config, preprocessing, feature_config)
            for name in self.config.classifiers:
                scaler = None
                train_values, val_values = x_train[:, columns], x_val[:, columns]
                if name == "svm":
                    scaler = StandardScaler().fit(train_values)
                    train_values = scaler.transform(train_values)
                    val_values = scaler.transform(val_values)
                for params in expand_grid(self.grids[name]):
                    model = make_classifier(
                        name, params, self.config.seed, self.config.balanced, self.config.n_jobs
                    )
                    start = time.perf_counter()
                    model.fit(train_values, y_train)
                    seconds = time.perf_counter() - start
                    train_metrics = score_predictions(y_train, model.predict(train_values))
                    validation_metrics = score_predictions(y_val, model.predict(val_values))
                    trial = {
                        "trial": len(trials) + 1, "feature_set": feature_set,
                        "classifier": name, "params": json.dumps(to_python(params), sort_keys=True),
                        "n_features": int(len(columns)),
                        "train_accuracy": round(train_metrics["accuracy"], 6),
                        "validation_accuracy": round(validation_metrics["accuracy"], 6),
                        "validation_balanced_accuracy": round(validation_metrics["balanced_accuracy"], 6),
                        "validation_macro_f1": round(validation_metrics["macro_f1"], 6),
                        "fit_seconds": round(seconds, 4),
                    }
                    trials.append(trial)
                    self.log(
                        f"[{trial['trial']}] {feature_set} + {name} {trial['params']}: "
                        f"val macro-F1={trial['validation_macro_f1']:.3f}, "
                        f"val accuracy={trial['validation_accuracy']:.3f}"
                    )
                    if best is None or selection_key(validation_metrics) > selection_key(best["validation"]):
                        best = {
                            "trial": trial, "feature_set": feature_set, "classifier": name,
                            "params": to_python(params), "model": model, "scaler": scaler,
                            "train": train_metrics, "validation": validation_metrics,
                            "fit_val": val_values, "columns": columns,
                        }
        return trials, best

    def _build_pipeline(self, preprocessing, feature_config, best):
        steps = [("features", ImageFeatureExtractor(preprocessing, feature_config))]
        if best["scaler"] is not None:
            steps.append(("scale", best["scaler"]))
        steps.append(("classifier", best["model"]))
        return Pipeline(steps)

    def _verify_pipeline(self, pipeline, data, x_val, best):
        rows = data["by_split"]["validation"][:5]
        paths = [resolve_image_path(data["dataset_dir"], row["relative_path"]) for row in rows]
        actual = pipeline.named_steps["features"].transform(paths)
        expected = x_val[:len(rows)][:, best["columns"]]
        if not np.allclose(actual, expected, atol=1e-6):
            raise TrainingError("Pipeline features differ from cached training features.")
        expected_predictions = best["model"].predict(best["fit_val"][:len(rows)])
        if not np.array_equal(pipeline.predict(paths), expected_predictions):
            raise TrainingError("Pipeline predictions differ from the selected model.")

    def _build_reports(self, data, preprocessing, selected_configs, trials, best):
        metadata = {
            "model_file": "pipeline.joblib", "class_labels": {"0": "outside", "1": "inside"},
            "feature_set": best["feature_set"],
            "feature_config": selected_configs[best["feature_set"]].to_dict(),
            "preprocessing_config": preprocessing.to_dict(),
            "classifier": best["classifier"], "classifier_params": best["params"],
            "class_weight": "balanced" if self.config.balanced else None,
            "estimator_class_weight": best["model"].class_weight,
            "seed": self.config.seed, "selection_rule": SELECTION_RULE,
            "trained_on_splits": ["train"], "selected_using_splits": ["validation"],
            "test_images_opened_by_training": 0,
            "images_csv_sha256": data["images_csv_sha256"], "counts": count_summary(data),
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
                "Pillow": PIL.__version__, "joblib": joblib.__version__,
            },
            "development_run": bool(self.config.allow_unverified),
        }
        remaining = [trial for trial in trials if trial["trial"] != best["trial"]["trial"]]
        remaining.sort(key=lambda trial: (
            -trial["validation_macro_f1"], -trial["validation_balanced_accuracy"],
            -trial["validation_accuracy"], trial["trial"],
        ))
        report = {
            **metadata, "best_trial": best["trial"], "train_metrics": best["train"],
            "validation_metrics": best["validation"],
            "top_trials": [best["trial"], *remaining[:4]], "trial_count": len(trials),
            "warning": "Validation was used for selection; report final generalisation only from the untouched test split.",
        }
        return metadata, report
