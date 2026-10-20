import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from polyu_lr_checks.hog_test_transformer import HogTestTransformer
from src.experiments.polyu_lr_config import PolyuLRConfig


def fixture(root):
    root = Path(root)
    dataset = root / "dataset"
    dataset.mkdir()
    images, locations = [], []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            group = f"{split}_{label}"
            locations.append({"group_id": group, "class_name": "inside" if label else "outside", "boundary_verified": "yes"})
            for index in range(4 if split == "train" else 2):
                relative = f"{group}_{index}.png"
                if split == "test":
                    (dataset / relative).write_bytes(b"intentionally invalid image")
                else:
                    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
                    stripe = (np.arange(64) // 8 % 2) * 220
                    pixels[:] = stripe[None, :, None] if label else stripe[:, None, None]
                    Image.fromarray(pixels).save(dataset / relative)
                images.append({"image_id": relative, "relative_path": relative, "label": str(label), "group_id": group, "split": split})
    for name, rows in (("images.csv", images), ("locations.csv", locations)):
        with (dataset / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    by_split = {name: [row for row in images if row["split"] == name] for name in ("train", "validation", "test")}
    paths = {name: [dataset / row["relative_path"] for row in by_split[name]] for name in ("train", "validation")}
    labels = {name: [int(row["label"]) for row in by_split[name]] for name in paths}
    baseline = Pipeline([("features", HogTestTransformer()), ("scale", StandardScaler()), ("classifier", SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=42))])
    baseline.fit(paths["train"], labels["train"])
    pred = baseline.predict(paths["validation"])
    metrics = {
        "accuracy": accuracy_score(labels["validation"], pred),
        "balanced_accuracy": balanced_accuracy_score(labels["validation"], pred),
        "macro_f1": f1_score(labels["validation"], pred, average="macro"),
    }
    metadata = {
        "images_csv_sha256": hashlib.sha256((dataset / "images.csv").read_bytes()).hexdigest(),
        "feature_set": "hog", "feature_config": {"use_color": False, "use_hog": True, "use_lbp": False, "hog_pixels_per_cell": 16},
        "preprocessing_config": {"image_size": 64, "resize_mode": "squash", "pad_value": 0},
        "classifier": "svm", "classifier_params": {"kernel": "linear", "C": 0.1},
        "seed": 42, "estimator_class_weight": "balanced", "class_weight": "balanced",
    }
    reference = {**metadata, "validation_metrics": metrics}
    metadata_path, report_path = root / "metadata.json", root / "reference.json"
    metadata_path.write_text(json.dumps(metadata))
    report_path.write_text(json.dumps(reference))
    config = PolyuLRConfig(dataset, metadata_path, report_path, root / "output")
    loader = lambda settings: {"dataset_dir": dataset, "rows": images, "by_split": by_split}
    transformer_factory = lambda settings: HogTestTransformer()
    return config, loader, transformer_factory
