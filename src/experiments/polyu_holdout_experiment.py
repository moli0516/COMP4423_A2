import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import PIL
import skimage
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from .polyu_pipeline_factory import PolyuPipelineFactory


class PolyuHoldoutExperiment:
    """Seven bounded fits on the frozen train/validation split; no test decoding."""

    def __init__(self, config, *, dataset_loader=None, transformer_factory=None, log=print):
        self.config = config
        self.dataset_loader = dataset_loader or self._load_dataset
        self.transformer_factory = transformer_factory or self._make_transformer
        self.log = log

    @staticmethod
    def _load_dataset(config):
        from src.data.dataset_catalog import DatasetCatalog
        return DatasetCatalog(config.dataset_dir, config.allow_unverified).load()

    @staticmethod
    def _make_transformer(metadata):
        from src.features import FeatureConfig, ImageFeatureExtractor
        from src.preprocessing import PreprocessingConfig
        return ImageFeatureExtractor(
            PreprocessingConfig.from_dict(metadata["preprocessing_config"]),
            FeatureConfig.from_dict(metadata["feature_config"]),
        )

    @staticmethod
    def _sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _python(value):
        if isinstance(value, dict):
            return {str(key): PolyuHoldoutExperiment._python(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [PolyuHoldoutExperiment._python(item) for item in value]
        if isinstance(value, np.ndarray):
            return PolyuHoldoutExperiment._python(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _json(cls, path, value):
        Path(path).write_text(json.dumps(cls._python(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def _csv(path, rows):
        if not rows:
            return
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _metrics(cls, y_true, y_pred):
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        return cls._python({
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
            "per_class": {
                name: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": support[i]}
                for i, name in enumerate(("outside", "inside"))
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
            "confusion_matrix_labels": ["outside(0)", "inside(1)"],
        })

    def preflight(self):
        config = self.config
        if config.output_dir.exists():
            raise FileExistsError(f"Output already exists; preserve it and choose a new directory: {config.output_dir}")
        metadata = json.loads(config.reference_metadata.read_text(encoding="utf-8"))
        reference = json.loads(config.reference_report.read_text(encoding="utf-8"))
        csv_hash = self._sha(config.dataset_dir / "images.csv")
        for name, record in (("metadata", metadata), ("report", reference)):
            if record.get("images_csv_sha256") != csv_hash:
                raise ValueError(f"Reference {name} does not match current images.csv; do not resplit or relabel artifacts.")
        if metadata.get("feature_set") != "hog":
            raise ValueError("The reference must be the current HOG-only baseline.")
        features = metadata["feature_config"]
        if features.get("use_color", True) or not features.get("use_hog", False) or features.get("use_lbp", False):
            raise ValueError("Use exactly the HOG-only feature configuration.")
        if metadata.get("classifier") != "svm" or metadata.get("classifier_params", {}).get("kernel") != "linear" or metadata["classifier_params"].get("C") != 0.1:
            raise ValueError("Reference must be Linear SVM C=0.1.")
        if metadata.get("seed") != config.seed:
            raise ValueError("Experiment seed must match the reference seed.")
        for key in ("feature_set", "feature_config", "preprocessing_config", "classifier", "classifier_params", "seed"):
            if reference.get(key) != metadata.get(key):
                raise ValueError(f"Reference report and metadata disagree on {key}.")
        weight = metadata.get("estimator_class_weight", metadata.get("class_weight"))
        if weight not in (None, "balanced"):
            raise ValueError("Expected reference class_weight to be balanced or null.")
        data = self.dataset_loader(config)
        by_split = data["by_split"]
        groups = {name: {r["group_id"] for r in rows} for name, rows in by_split.items()}
        for name in ("train", "validation", "test"):
            if name not in by_split or {str(r["label"]) for r in by_split[name]} != {"0", "1"}:
                raise ValueError(f"Both classes are required in {name}.")
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[left] & groups[right]:
                raise ValueError("Location group leakage between splits.")
        for name in ("train", "validation"):
            for row in by_split[name]:
                path = (config.dataset_dir / row["relative_path"]).resolve()
                if not path.is_relative_to(config.dataset_dir):
                    raise ValueError("Image path escapes dataset directory.")
                if not path.is_file():
                    raise FileNotFoundError(path)
        transformer = self.transformer_factory(metadata)
        factory = PolyuPipelineFactory(config, weight)
        manifest = {
            "protocol": "POLYU-LR-01 fixed grouped holdout, six LR candidates plus a freshly fitted SVM baseline",
            "config": config.to_dict(), "images_csv_sha256": csv_hash,
            "locations_csv_sha256": self._sha(config.dataset_dir / "locations.csv"),
            "reference_metadata_sha256": self._sha(config.reference_metadata),
            "reference_report_sha256": self._sha(config.reference_report),
            "candidates": list(factory.candidates()),
            "selection_rule": "eligible macro-F1, balanced accuracy, accuracy; earlier candidate wins exact ties",
            "eligibility_rule": "finite outputs; no ConvergenceWarning; iterations strictly below max_iter when limited",
            "counts": {name: {"images": len(rows), "groups": len(groups[name])} for name, rows in by_split.items()},
            "test_images_opened": 0,
            "evaluation_status": "development holdout used for model selection; not independent final evaluation",
        }
        return data, metadata, reference, transformer, factory, manifest

    def run(self, dry_run=False):
        data, metadata, reference, transformer, factory, manifest = self.preflight()
        if dry_run:
            self.log(json.dumps(self._python(manifest), indent=2))
            self.log("DRY RUN: no image decoding, fitting or output-directory creation.")
            return {"dry_run": True, "manifest": manifest}
        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=False)
        manifest["status"] = "running"
        source_root = Path(__file__).resolve().parent
        manifest["source_sha256"] = {path.name: self._sha(path) for path in sorted(source_root.glob("*.py"))}
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, timeout=5)
            manifest["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            manifest["git_revision"] = None
        self._json(output / "config.json", self.config.to_dict())
        self._json(output / "run_manifest.json", manifest)
        self._json(output / "environment.json", {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "Pillow": PIL.__version__,
            "scikit_image": skimage.__version__, "joblib": joblib.__version__,
        })
        results, table, best = [], [], None
        try:
            records = [dict(row) for rows in data["by_split"].values() for row in rows]
            self._json(output / "dataset_records.json", records)
            paths = {name: [(self.config.dataset_dir / row["relative_path"]).resolve() for row in data["by_split"][name]] for name in ("train", "validation")}
            labels = {name: np.array([int(row["label"]) for row in data["by_split"][name]]) for name in paths}
            image_hashes = {str(path.relative_to(self.config.dataset_dir)): self._sha(path) for subset in paths.values() for path in subset}
            self._json(output / "train_validation_image_hashes.json", image_hashes)
            transformer.fit(paths["train"], labels["train"])
            matrices = {name: np.asarray(transformer.transform(subset), dtype=np.float32) for name, subset in paths.items()}
            if any(matrix.ndim != 2 or not np.isfinite(matrix).all() for matrix in matrices.values()):
                raise ValueError("Invalid or non-finite feature matrix.")
            for index, candidate in enumerate(factory.candidates()):
                manifest["current_candidate"] = candidate["id"]
                self._json(output / "run_manifest.json", manifest)
                directory = output / "candidates" / candidate["id"]
                directory.mkdir(parents=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline = factory.create(candidate, transformer)
                    pipeline.named_steps["features"].fit(paths["train"], labels["train"])
                    scaler = pipeline.named_steps["scale"]
                    x_train = scaler.fit_transform(matrices["train"])
                    x_valid = scaler.transform(matrices["validation"])
                    classifier = pipeline.named_steps["classifier"]
                    start = time.perf_counter()
                    classifier.fit(x_train, labels["train"])
                    seconds = time.perf_counter() - start
                train_metrics = self._metrics(labels["train"], classifier.predict(x_train))
                validation_pred = classifier.predict(x_valid)
                validation_metrics = self._metrics(labels["validation"], validation_pred)
                warning_records = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                iterations = np.asarray(getattr(classifier, "n_iter_", []), dtype=int)
                limit = classifier.get_params().get("max_iter", -1)
                limit_hit = limit > 0 and bool(np.any(iterations >= limit))
                coefficients = np.asarray(classifier.coef_)
                if not np.isfinite(coefficients).all():
                    raise ValueError("Non-finite model coefficients.")
                diagnostics = {
                    "warnings": warning_records, "convergence_warning": convergence,
                    "n_iter": iterations.tolist(), "iteration_limit_hit": limit_hit,
                    "exact_nonzero_count": int(np.count_nonzero(coefficients)),
                    "effective_nonzero_count": int(np.count_nonzero(np.abs(coefficients) > self.config.effective_zero_threshold)),
                    "effective_zero_threshold": self.config.effective_zero_threshold,
                    "n_features": int(matrices["train"].shape[1]), "fit_seconds": seconds,
                    "eligible": not convergence and not limit_hit,
                    "estimator_params": classifier.get_params(deep=False),
                }
                if candidate["family"] == "svm":
                    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
                        if abs(validation_metrics[key] - reference["validation_metrics"][key]) > 1e-10:
                            raise ValueError("Fresh SVM baseline does not reproduce the reference. Inspect data, code and settings; do not relabel artifacts.")
                uncached_pred = pipeline.predict(paths["validation"])
                if not np.array_equal(uncached_pred, validation_pred):
                    raise ValueError("Saved-pipeline inference differs from cached-feature inference.")
                scores = np.asarray(classifier.decision_function(x_valid))
                if scores.shape != (len(validation_pred),) or not np.isfinite(scores).all():
                    raise ValueError("Invalid binary decision scores.")
                prediction_rows = [
                    {"image_id": row["image_id"], "relative_path": row["relative_path"], "group_id": row["group_id"], "true_label": int(row["label"]), "predicted_label": int(pred), "decision_score_for_inside": float(score)}
                    for row, pred, score in zip(data["by_split"]["validation"], validation_pred, scores)
                ]
                metrics = {"train_metrics": train_metrics, "validation_metrics": validation_metrics}
                self._json(directory / "config.json", candidate)
                self._json(directory / "metrics.json", metrics)
                self._json(directory / "diagnostics.json", diagnostics)
                self._csv(directory / "validation_predictions.csv", prediction_rows)
                np.savez_compressed(directory / "coefficients.npz", coefficients=coefficients, intercept=classifier.intercept_)
                joblib.dump(pipeline, directory / "pipeline.joblib")
                reloaded = joblib.load(directory / "pipeline.joblib")
                if not np.array_equal(reloaded.predict(paths["validation"]), validation_pred):
                    raise ValueError("Reloaded pipeline predictions differ.")
                self._json(directory / "pipeline_metadata.json", {
                    "candidate": candidate, "class_labels": {"0": "outside", "1": "inside"},
                    "feature_set": "hog", "feature_config": metadata["feature_config"],
                    "preprocessing_config": metadata["preprocessing_config"],
                    "images_csv_sha256": manifest["images_csv_sha256"],
                    "trained_on_splits": ["train"], "selected_using_splits": ["validation"],
                    "test_images_opened_by_training": 0, "development_run": self.config.allow_unverified,
                    "eligible": diagnostics["eligible"], "estimator_params": classifier.get_params(deep=False),
                })
                row = {
                    "candidate_id": candidate["id"], "family": candidate["family"],
                    "penalty": candidate["penalty"], "C": candidate["C"],
                    "eligible": diagnostics["eligible"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "outside_recall": validation_metrics["per_class"]["outside"]["recall"],
                    "inside_recall": validation_metrics["per_class"]["inside"]["recall"],
                    "inside_precision": validation_metrics["per_class"]["inside"]["precision"],
                    "effective_nonzero_count": diagnostics["effective_nonzero_count"],
                    "warning_count": len(caught), "fit_seconds": seconds,
                }
                baseline = row if index == 0 else table[0]
                row["macro_f1_difference_vs_baseline"] = row["validation_macro_f1"] - baseline["validation_macro_f1"]
                table.append(row)
                result = {"candidate": candidate, **metrics, "diagnostics": diagnostics}
                results.append(result)
                self._csv(output / "comparison.csv", table)
                self._json(output / "comparison.json", results)
                key = tuple(row[name] for name in ("validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"))
                if diagnostics["eligible"] and (best is None or key > best["key"]):
                    best = {"key": key, "candidate_id": candidate["id"], "result": result}
                self.log(f"{candidate['id']}: macro-F1={validation_metrics['macro_f1']:.4f}, outside recall={row['outside_recall']:.4f}, eligible={diagnostics['eligible']}")
            if len(table) != 7:
                raise RuntimeError("The frozen experiment requires exactly seven completed fits.")
            if self._sha(self.config.dataset_dir / "images.csv") != manifest["images_csv_sha256"] or self._sha(self.config.dataset_dir / "locations.csv") != manifest["locations_csv_sha256"]:
                raise ValueError("Dataset metadata changed during the experiment.")
            if any(self._sha(self.config.dataset_dir / relative) != digest for relative, digest in image_hashes.items()):
                raise ValueError("Image content changed during the experiment.")
            if best is None:
                raise RuntimeError("No numerically eligible candidate. Inspect diagnostics without expanding the search.")
            best_dir = output / "best"
            best_dir.mkdir()
            for name in ("pipeline.joblib", "pipeline_metadata.json"):
                shutil.copy2(output / "candidates" / best["candidate_id"] / name, best_dir / name)
            self._json(output / "best_result.json", best["result"])
            manifest.update(status="completed", completed_fits=7, best_candidate=best["candidate_id"], numerical_review_required=any(not row["eligible"] for row in table))
            self._json(output / "run_manifest.json", manifest)
            self.log(f"Best eligible candidate: {best['candidate_id']}")
            self.log("Development comparison only. The held-out test images were not opened.")
            return {"manifest": manifest, "comparison": table, "best": best["result"]}
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", completed_fits=len(table), error=f"{type(error).__name__}: {error}")
            self._json(output / "run_manifest.json", manifest)
            raise
