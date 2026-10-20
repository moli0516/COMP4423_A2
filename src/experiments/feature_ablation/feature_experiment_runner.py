import csv
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from .feature_experiment_planner import json_safe, sha256_file
from .feature_pipeline_factory import FeaturePipelineFactory
from .feature_variant import FeatureVariant


class FeatureExperimentRunner:
    """Execute a frozen POLYU-FEATURE-01 plan; never creates or edits the plan."""

    RECORD_FIELDS = ("image_id", "relative_path", "label", "group_id", "split")

    def __init__(self, config, output_dir, *, camera_diagnostic_confirmed=False,
                 dimension_function=None, log=print):
        if type(camera_diagnostic_confirmed) is not bool:
            raise ValueError("camera_diagnostic_confirmed must be a boolean.")
        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.camera_diagnostic_confirmed = camera_diagnostic_confirmed
        self.dimension_function = dimension_function
        self.log = log

    # ----- small file helpers -------------------------------------------------
    @staticmethod
    def _read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write(path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=path.parent,
                prefix=".run_", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(text)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def _write_json(cls, path, value):
        cls._atomic_write(path, json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    @classmethod
    def _write_csv(cls, path, rows):
        if not rows:
            return
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        cls._atomic_write(path, buffer.getvalue())

    @staticmethod
    def _metrics(truth, predictions):
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predictions, labels=[0, 1], zero_division=0)
        return json_safe({
            "accuracy": accuracy_score(truth, predictions),
            "balanced_accuracy": balanced_accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0),
            "per_class": {
                name: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": support[i]}
                for i, name in enumerate(("outside", "inside"))
            },
            "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]),
            "confusion_matrix_labels": ["outside(0)", "inside(1)"],
        })

    # ----- protocol verification ---------------------------------------------
    def _inputs(self):
        config = self.config
        return {
            "images_csv": config.dataset_dir / "images.csv",
            "locations_csv": config.dataset_dir / "locations.csv",
            "reference_model": config.reference_model,
            "reference_metadata": config.reference_metadata,
            "reference_result": config.reference_result,
        }

    def _check_output_location(self):
        output, config = self.output_dir, self.config
        if output.exists():
            raise FileExistsError(f"Output already exists. Preserve it and choose a new directory: {output}")
        protected = {
            "dataset": config.dataset_dir, "plan": config.plan_dir,
            "reference model directory": config.reference_model.parent,
        }
        for name, path in protected.items():
            if output == path or path in output.parents or output in path.parents:
                raise ValueError(f"Output must be separate from the {name}: {path}")

    def _load_verified_plan(self):
        config = self.config
        plan_path = config.plan_dir / "experiment_plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(
                f"Frozen plan not found: {plan_path}. Run scripts/prepare_polyu_feature_experiment.py --freeze first.")
        plan = self._read_json(plan_path)
        for key, expected in (
            ("experiment_id", config.experiment_id), ("stage", "prepared_architecture_only"),
            ("training_implemented", False), ("completed_fits", 0), ("planned_fits", 4),
        ):
            if plan.get(key) != expected or type(plan.get(key)) is not type(expected):
                raise ValueError(f"Unexpected plan field {key}: {plan.get(key)!r}")
        current, frozen = config.to_dict(), plan.get("config", {})
        if set(current) != set(frozen):
            raise ValueError("Current configuration fields differ from the frozen plan.")
        for key, value in current.items():
            if key != "camera_diagnostic_status" and frozen[key] != value:
                raise ValueError(f"Current configuration differs from the frozen plan: {key}")
        if self._read_json(config.plan_dir / "config.json") != frozen:
            raise ValueError("plan/config.json differs from experiment_plan.json.")
        planned = FeatureVariant.planned_variants()
        frozen_variants = plan.get("variants", [])
        if [item.get("variant_id") for item in frozen_variants] != [item.variant_id for item in planned]:
            raise ValueError("Frozen plan does not contain exactly F0, F1, F2 and F3.")
        for variant, record in zip(planned, frozen_variants):
            expected = variant.to_dict()
            if {key: record.get(key) for key in expected} != expected:
                raise ValueError(f"Frozen variant {variant.variant_id} differs from FeatureVariant.")
            if self._read_json(config.plan_dir / "variants" / f"{variant.variant_id}.json") != record:
                raise ValueError(f"Variant file differs from the plan: {variant.variant_id}")
        input_hashes = {}
        for name, path in self._inputs().items():
            item = plan.get("input_files", {}).get(name)
            if item is None or Path(item["path"]).resolve() != path:
                raise ValueError(f"Plan input path differs from current configuration: {name}")
            digest = sha256_file(path)
            if digest != item["sha256"]:
                raise ValueError(f"Input changed since the plan was frozen: {name}")
            input_hashes[name] = digest
        snapshots = {"images_csv": config.plan_dir / "snapshots/images.csv",
                     "locations_csv": config.plan_dir / "snapshots/locations.csv"}
        for name, path in snapshots.items():
            if sha256_file(path) != plan["input_files"][name]["sha256"]:
                raise ValueError(f"Plan snapshot differs from its frozen digest: {name}")
        with snapshots["images_csv"].open(encoding="utf-8-sig", newline="") as handle:
            snapshot_rows = list(csv.DictReader(handle))
        snapshot = {row["image_id"].strip(): {name: row[name].strip() for name in self.RECORD_FIELDS}
                    for row in snapshot_rows}
        records = plan.get("dataset_records", [])
        frozen_records = {str(row["image_id"]): {name: str(row[name]) for name in self.RECORD_FIELDS} for row in records}
        if len(snapshot) != len(snapshot_rows) or len(frozen_records) != len(records) or snapshot != frozen_records:
            raise ValueError("Frozen dataset records differ from the images.csv snapshot.")
        subsets = {name: [row for row in records if row["split"] == name] for name in ("train", "validation", "test")}
        if sum(len(rows) for rows in subsets.values()) != len(records):
            raise ValueError("Frozen records contain an unknown split.")
        groups = {}
        for name, rows in subsets.items():
            if {str(row["label"]) for row in rows} != {"0", "1"}:
                raise ValueError(f"Both classes are required in {name}.")
            groups[name] = {row["group_id"] for row in rows}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[left] & groups[right]:
                raise ValueError(f"Location group leakage between {left} and {right}.")
        expected_hashes = plan["train_validation_image_hashes"]
        wanted = set()
        for name in ("train", "validation"):
            for row in subsets[name]:
                path = (config.dataset_dir / row["relative_path"]).resolve()
                if not path.is_relative_to(config.dataset_dir):
                    raise ValueError("Image path escapes the dataset directory.")
                wanted.add(path.relative_to(config.dataset_dir).as_posix())
        if wanted != set(expected_hashes):
            raise ValueError("Frozen image hashes do not match the train/validation records.")
        self._verify_images(expected_hashes)
        reference = joblib.load(config.reference_model)
        factory = FeaturePipelineFactory(reference, self.dimension_function)
        for variant, record in zip(planned, frozen_variants):
            if json_safe(factory.describe(variant)) != record:
                raise ValueError(f"Frozen protocol for {variant.variant_id} differs from the current reference pipeline.")
        metadata = self._read_json(config.reference_metadata)
        result = self._read_json(config.reference_result)
        if not (metadata.get("candidate") == plan["reference_candidate"] == result.get("candidate")):
            raise ValueError("Reference candidate records disagree with the plan.")
        if json_safe(result.get("validation_metrics")) != plan["reference_validation_metrics"]:
            raise ValueError("Reference validation metrics differ from the frozen plan.")
        return {
            "plan": plan, "plan_sha256": sha256_file(plan_path), "plan_path": plan_path,
            "subsets": subsets, "variants": planned, "factory": factory, "reference": reference,
            "input_hashes": input_hashes, "image_hashes": expected_hashes,
        }

    def _verify_images(self, expected_hashes):
        for relative, digest in expected_hashes.items():
            path = (self.config.dataset_dir / relative).resolve()
            if not path.is_relative_to(self.config.dataset_dir):
                raise ValueError("Image path escapes the dataset directory.")
            if sha256_file(path) != digest:
                raise ValueError(f"Train/validation image changed: {relative}")

    def _gate(self):
        status = self.config.camera_diagnostic_status
        return {
            "config_status": status, "cli_confirmation": self.camera_diagnostic_confirmed,
            "open": status == "completed" or self.camera_diagnostic_confirmed,
        }

    # ----- public entry -------------------------------------------------------
    def run(self, dry_run=False):
        self._check_output_location()
        context = self._load_verified_plan()
        gate = self._gate()
        summary = {
            "variants": [(item["variant_id"], item["feature_dimension"]) for item in context["plan"]["variants"]],
            "train_images": len(context["subsets"]["train"]), "validation_images": len(context["subsets"]["validation"]),
            "test_images_hashed_or_decoded": 0, "camera_gate": gate,
        }
        if dry_run:
            for variant_id, dimension in summary["variants"]:
                self.log(f"{variant_id}: dimension={dimension}")
            self.log(f"Train/validation images: {summary['train_images']}/{summary['validation_images']}")
            self.log(f"Camera diagnostic gate: {'open' if gate['open'] else 'closed'}")
            self.log("DRY RUN: plan, input hashes and reference protocol verified; no image decoding, fitting or output creation.")
            return {"dry_run": True, "summary": summary}
        if not gate["open"]:
            raise PermissionError(
                "camera_diagnostic_status is pending. Run the saved-image versus camera-predictor diagnostic, "
                "then pass --confirm-camera-diagnostic-completed (or set the config status to completed).")
        return self._execute(context, gate)

    def _execute(self, context, gate):
        output, config = self.output_dir, self.config
        plan, factory, reference = context["plan"], context["factory"], context["reference"]
        output.mkdir(parents=True, exist_ok=False)
        source_root = Path(__file__).resolve().parent
        manifest = {
            "experiment_id": config.experiment_id, "stage": "runner_execution", "status": "running",
            "training_implemented": True, "planned_fits": 4, "completed_fits": 0,
            "plan_dir": str(config.plan_dir), "plan_sha256": context["plan_sha256"],
            "input_hashes": context["input_hashes"], "camera_gate": gate,
            "selection_rule": plan["rules"]["selection"], "test_images_hashed_or_decoded": 0,
            "evaluation_status": "development model selection, not independent final evaluation",
            "source_sha256": {path.name: sha256_file(path) for path in sorted(source_root.glob("*.py"))},
        }
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, timeout=5)
            manifest["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            manifest["git_revision"] = None
        self._write_json(output / "run_manifest.json", manifest)
        self._write_json(output / "config.json", {**config.to_dict(), "output_dir": str(output)})
        self._write_json(output / "environment.json", {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__})
        table, results, best, f0 = [], [], None, None
        try:
            subsets = context["subsets"]
            dataset = config.dataset_dir
            paths = {name: [(dataset / row["relative_path"]).resolve() for row in subsets[name]] for name in ("train", "validation")}
            labels = {name: np.array([int(row["label"]) for row in subsets[name]]) for name in paths}
            reference_predictions = np.asarray(reference.predict(paths["validation"]))
            for variant in context["variants"]:
                manifest["current_variant"] = variant.variant_id
                self._write_json(output / "run_manifest.json", manifest)
                described = next(item for item in plan["variants"] if item["variant_id"] == variant.variant_id)
                dimension = described["feature_dimension"]
                pipeline = factory.create(variant)
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(paths["train"], labels["train"])
                    predictions = np.asarray(pipeline.predict(paths["validation"]))
                    scores = np.asarray(pipeline.decision_function(paths["validation"]))
                seconds = time.perf_counter() - start
                classifier = pipeline.named_steps["classifier"]
                coefficients = np.asarray(classifier.coef_)
                if coefficients.shape != (1, dimension) or not np.isfinite(coefficients).all():
                    raise ValueError(f"{variant.variant_id}: invalid coefficient shape or values.")
                if scores.shape != (len(predictions),) or not np.isfinite(scores).all():
                    raise ValueError(f"{variant.variant_id}: invalid decision scores.")
                iterations = np.asarray(classifier.n_iter_, dtype=int)
                convergence = any(issubclass(item.category, ConvergenceWarning) for item in caught)
                limit = int(classifier.get_params()["max_iter"])
                limit_hit = bool(np.any(iterations >= limit))
                eligible = not convergence and not limit_hit
                metrics = self._metrics(labels["validation"], predictions)
                if variant.variant_id == "F0":
                    if not eligible:
                        raise ValueError("F0 failed numerical eligibility; do not compare alternatives against an invalid control.")
                    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
                        if abs(metrics[key] - plan["reference_validation_metrics"][key]) > 1e-10:
                            raise ValueError("F0 does not reproduce the reference metrics. Inspect data, code and settings; do not rewrite the baseline.")
                    if not np.array_equal(predictions, reference_predictions):
                        raise ValueError("F0 predictions differ from the saved reference model.")
                    f0 = {"metrics": metrics, "predictions": predictions}
                directory = output / "candidates" / variant.variant_id
                directory.mkdir(parents=True)
                joblib.dump(pipeline, directory / "pipeline.joblib")
                reloaded = joblib.load(directory / "pipeline.joblib")
                if not np.array_equal(reloaded.predict(paths["validation"]), predictions):
                    raise ValueError(f"{variant.variant_id}: reloaded pipeline predictions differ.")
                images = []
                for path in paths["validation"]:
                    with Image.open(path) as opened:
                        opened.load()
                        images.append(opened.copy())
                if not np.array_equal(np.asarray(reloaded.predict(images)), predictions):
                    raise ValueError(f"{variant.variant_id}: PIL-image inference differs from file-path inference.")
                diagnostics = {
                    "eligible": eligible, "convergence_warning": convergence, "iteration_limit_hit": limit_hit,
                    "n_iter": iterations, "max_iter": limit, "fit_and_predict_seconds": seconds,
                    "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
                    "exact_nonzero_count": int(np.count_nonzero(coefficients)),
                    "estimator_params": classifier.get_params(deep=False),
                }
                rows = [
                    {"image_id": row["image_id"], "relative_path": row["relative_path"], "group_id": row["group_id"],
                     "true_label": int(row["label"]), "predicted_label": int(prediction),
                     "decision_score_for_inside": float(score), "reference_predicted_label": int(reference_label)}
                    for row, prediction, score, reference_label in zip(subsets["validation"], predictions, scores, reference_predictions)
                ]
                metadata = {
                    "experiment_id": config.experiment_id, "variant": variant.to_dict(),
                    "class_labels": {"0": "outside", "1": "inside"},
                    "feature_config": described["feature_config"], "preprocessing_config": described["preprocessing_config"],
                    "feature_dimension": dimension, "plan_sha256": context["plan_sha256"],
                    "images_csv_sha256": context["input_hashes"]["images_csv"],
                    "trained_on_splits": ["train"], "selected_using_splits": ["validation"],
                    "test_images_opened_by_training": 0, "development_run": config.allow_unverified,
                    "eligible": eligible, "estimator_params": classifier.get_params(deep=False),
                }
                self._write_json(directory / "variant.json", described)
                self._write_json(directory / "metrics.json", metrics)
                self._write_json(directory / "diagnostics.json", diagnostics)
                self._write_json(directory / "pipeline_metadata.json", metadata)
                self._write_csv(directory / "validation_predictions.csv", rows)
                np.savez_compressed(directory / "coefficients.npz", coefficients=coefficients, intercept=classifier.intercept_)
                row = {
                    "variant_id": variant.variant_id, "description": variant.description, "n_features": dimension,
                    "eligible": eligible, "validation_macro_f1": metrics["macro_f1"],
                    "validation_balanced_accuracy": metrics["balanced_accuracy"], "validation_accuracy": metrics["accuracy"],
                    "outside_recall": metrics["per_class"]["outside"]["recall"],
                    "inside_recall": metrics["per_class"]["inside"]["recall"],
                    "inside_precision": metrics["per_class"]["inside"]["precision"],
                    "macro_f1_difference_vs_f0": metrics["macro_f1"] - f0["metrics"]["macro_f1"],
                    "outside_recall_difference_vs_f0": metrics["per_class"]["outside"]["recall"] - f0["metrics"]["per_class"]["outside"]["recall"],
                    "predictions_changed_vs_f0": int(np.sum(predictions != f0["predictions"])),
                    "warning_count": len(caught), "fit_and_predict_seconds": seconds,
                }
                table.append(row)
                results.append({"variant": described, "metrics": metrics, "diagnostics": diagnostics})
                manifest["completed_fits"] = len(table)
                self._write_csv(output / "comparison.csv", table)
                self._write_json(output / "comparison.json", results)
                self._write_json(output / "run_manifest.json", manifest)
                key = (metrics["macro_f1"], metrics["balanced_accuracy"], metrics["accuracy"])
                if eligible and (best is None or key > best["key"]):
                    best = {"key": key, "variant_id": variant.variant_id, "result": results[-1]}
                self.log(f"{variant.variant_id}: dimension={dimension}, macro-F1={row['validation_macro_f1']:.4f}, "
                         f"outside recall={row['outside_recall']:.4f}, eligible={eligible}")
            if len(table) != 4 or best is None:
                raise RuntimeError("Expected four completed variants and at least one eligible candidate.")
            for name, path in self._inputs().items():
                if sha256_file(path) != context["input_hashes"][name]:
                    raise ValueError(f"Input changed during execution: {name}")
            if sha256_file(context["plan_path"]) != context["plan_sha256"]:
                raise ValueError("The frozen plan changed during execution.")
            self._verify_images(context["image_hashes"])
            best_directory = output / "best"
            best_directory.mkdir()
            for name in ("pipeline.joblib", "pipeline_metadata.json"):
                shutil.copy2(output / "candidates" / best["variant_id"] / name, best_directory / name)
            self._write_json(output / "best_result.json", best["result"])
            manifest.update(
                status="completed", best_variant=best["variant_id"],
                numerical_review_required=any(not item["eligible"] for item in table))
            manifest.pop("current_variant", None)
            self._write_json(output / "run_manifest.json", manifest)
            self.log(f"Best eligible development variant: {best['variant_id']}")
            self.log("No test images were hashed or decoded. This does not replace the camera diagnostic.")
            return {"manifest": manifest, "comparison": table, "best": best["result"]}
        except BaseException as error:
            manifest.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                completed_fits=len(table), error=f"{type(error).__name__}: {error}")
            self._write_json(output / "run_manifest.json", manifest)
            raise
