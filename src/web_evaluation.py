"""Evaluate the active pipeline on one dataset split for the web UI (Evaluate page).

The held-out test split is protected: it is only evaluated after an explicit confirmation, and every
fresh test evaluation is appended to ``.test_evaluations.json`` next to the models, so repeated
peeking at the test set is visible instead of silent.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

SPLITS = ("train", "validation", "test")
CLASS_NAMES = {0: "outside", 1: "inside"}
LOG_NAME = ".test_evaluations.json"


class UnknownImageError(LookupError):
    """The requested image is not part of the requested split."""


def read_split_rows(dataset_dir, split: str) -> list[dict]:
    images_csv = Path(dataset_dir) / "images.csv"
    if not images_csv.is_file():
        raise ValueError(f"{images_csv} was not found.")
    with images_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if (row.get("split") or "").strip() == split]


def resolve_image(dataset_dir, relative_path: str) -> Path:
    root = Path(dataset_dir).resolve()
    path = (root / str(relative_path).replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"Image path leaves the dataset directory: {relative_path}") from None
    return path


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Confusion matrix rows are the true class (0 outside, 1 inside); columns are the prediction."""
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(y_true, y_pred):
        matrix[int(truth)][int(guess)] += 1
    total = sum(sum(row) for row in matrix)
    per_class = {}
    for label, name in CLASS_NAMES.items():
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = matrix[label][1 - label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    present = [value["recall"] for value in per_class.values() if value["support"] > 0]
    return {
        "images": total,
        "correct": matrix[0][0] + matrix[1][1],
        "accuracy": (matrix[0][0] + matrix[1][1]) / total if total else 0.0,
        "balanced_accuracy": sum(present) / len(present) if present else 0.0,
        "macro_f1": sum(value["f1"] for value in per_class.values()) / 2,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def group_table(items: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for item in items:
        entry = groups.setdefault(item["group_id"], {"group_id": item["group_id"], "class": CLASS_NAMES[item["label"]], "images": 0, "correct": 0})
        entry["images"] += 1
        entry["correct"] += int(item["correct"])
    rows = list(groups.values())
    for row in rows:
        row["accuracy"] = row["correct"] / row["images"]
    return sorted(rows, key=lambda row: (row["accuracy"], row["group_id"]))


def margin_of(score, kind) -> float | None:
    if score is None:
        return None
    return abs(score - 0.5) if kind and "probability" in kind else abs(score)


def as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


@functools.lru_cache(maxsize=256)
def _thumbnail(path: str, mtime_ns: int, width: int) -> bytes:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((width, width))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
        return buffer.getvalue()


def make_thumbnail(path: Path, width: int) -> bytes:
    width = max(64, min(int(width), 2000))
    return _thumbnail(str(path), path.stat().st_mtime_ns, width)


class Evaluator:
    """Runs the active model over one split. ``store`` is the app's ModelStore."""

    def __init__(self, store, dataset_dir):
        self.store = store
        self.dataset_dir = Path(dataset_dir)
        self._lock = threading.Lock()
        self._cache: dict[tuple, dict] = {}

    @property
    def log_path(self) -> Path:
        return self.store.outputs_dir / LOG_NAME

    def _read_log(self) -> list[dict]:
        try:
            value = json.loads(self.log_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return value if isinstance(value, list) else []

    def _csv_digest(self) -> str:
        path = self.dataset_dir / "images.csv"
        if not path.is_file():
            raise ValueError(f"{path} was not found.")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_history(self, digest: str) -> list[dict]:
        return [entry for entry in self._read_log() if entry.get("images_csv_sha256") == digest]

    def evaluate(self, split: str, confirm_test: bool = False, refresh: bool = False) -> dict:
        if split not in SPLITS:
            raise ValueError("split must be train, validation or test.")
        model_id = self.store.active_id
        if model_id is None:
            raise LookupError("No trained pipeline found. Run training first.")
        if split == "test" and not confirm_test:
            raise PermissionError("The test split is held out. Tick the confirmation box to evaluate it.")
        digest = self._csv_digest()
        rows = read_split_rows(self.dataset_dir, split)
        if not rows:
            raise ValueError(f"The {split} split has no images in images.csv.")
        try:
            model_stamp = self.store._paths[model_id].stat().st_mtime_ns
        except (AttributeError, KeyError, OSError):
            model_stamp = 0
        key = (model_id, split, digest, model_stamp)
        with self._lock:
            if key in self._cache and not refresh:
                return {**self._cache[key], "cached": True}
            result = self._run(model_id, split, rows, digest)
            self._cache[key] = result
        if split == "test":
            self._log_test(result)
            result = {**result, "test_history": self.test_history(digest)}
            self._cache[key] = result
        return {**result, "cached": False}

    def _run(self, model_id: str, split: str, rows: list[dict], digest: str) -> dict:
        started = time.perf_counter()
        items, failures, kind = [], [], None
        for row in rows:
            image_id = row.get("image_id", "")
            try:
                data = resolve_image(self.dataset_dir, row["relative_path"]).read_bytes()
                prediction = self.store.predict(data, model_id)
                truth = int(str(row["label"]).strip())
                if truth not in CLASS_NAMES:
                    raise ValueError(f"label must be 0 or 1, found {row['label']!r}")
            except Exception as error:
                failures.append({"image_id": image_id, "group_id": row.get("group_id", ""), "error": str(error)})
                continue
            kind = kind or prediction.get("score_kind")
            score = prediction.get("score")
            items.append({
                "image_id": image_id, "group_id": row["group_id"], "label": truth, "pred": prediction["label"],
                "score": score, "margin": margin_of(score, prediction.get("score_kind")),
                "correct": truth == prediction["label"],
            })
        if not items:
            raise ValueError("No image in this split could be classified: " + (failures[0]["error"] if failures else "unknown error"))
        metrics = compute_metrics([item["label"] for item in items], [item["pred"] for item in items])
        meta = self.store.metadata(model_id)
        warnings = []
        if split in as_list(meta.get("trained_on_splits")):
            warnings.append(f"This model was trained on the {split} split, so these numbers are optimistic.")
        elif split in as_list(meta.get("selected_using_splits")):
            warnings.append(f"The {split} split was used to choose this model, so these numbers are slightly optimistic. Use the test split for the final estimate.")
        trained_digest = meta.get("images_csv_sha256")
        if trained_digest and trained_digest != digest:
            warnings.append("images.csv has changed since this model was trained. Retrain before trusting these numbers.")
        if failures:
            warnings.append(f"{len(failures)} image(s) could not be classified and are excluded from the metrics.")
        return {
            "split": split, "model": model_id, "images_csv_sha256": digest, "score_kind": kind,
            "metrics": metrics, "groups": group_table(items), "items": items, "failures": failures,
            "warnings": warnings, "elapsed_s": round(time.perf_counter() - started, 1),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _log_test(self, result: dict) -> None:
        entry = {
            "time": result["evaluated_at"], "model": result["model"], "images_csv_sha256": result["images_csv_sha256"],
            "accuracy": round(result["metrics"]["accuracy"], 4), "macro_f1": round(result["metrics"]["macro_f1"], 4),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(json.dumps(self._read_log() + [entry], indent=2), encoding="utf-8")
        except OSError:
            pass  # the log is an audit convenience only

    def image_file(self, split: str, image_id: str) -> Path:
        if split not in SPLITS:
            raise ValueError("split must be train, validation or test.")
        if split == "test" and not self.test_history(self._csv_digest()):
            raise PermissionError("Evaluate the test split first; its images stay locked until then.")
        for row in read_split_rows(self.dataset_dir, split):
            if row.get("image_id") == image_id:
                path = resolve_image(self.dataset_dir, row["relative_path"])
                if path.is_file():
                    return path
        raise UnknownImageError(image_id)
