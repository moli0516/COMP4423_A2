import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import joblib
import numpy as np
from PIL import Image
import src.train as training_cli
import src.storage.artifact_store as artifact_store_module
from src.features.feature_config import FeatureConfig
from src.preprocessing.preprocessing_config import PreprocessingConfig
from src.training.training_config import TrainingConfig
from src.training.model_trainer import ModelTrainer
from src.training.constants import DEFAULT_GRIDS
from src.training.search import expand_grid, selection_key
from src.training.metrics import score_predictions
from src.training.classifier_factory import make_classifier
from src.training.feature_matrix import feature_columns
from src.training.api import run_training
from src.data.api import load_dataset
from src.exceptions.training_error import TrainingError
from src.utils.file_io import resolve_image_path
from src.utils.serialization import to_python

SPLITS_BY_INDEX = {1: 'train', 2: 'train', 3: 'validation', 4: 'test'}
FAST_GRIDS = {'svm': [{'kernel': ['linear'], 'C': [1]}], 'random_forest': [{'n_estimators': [10], 'max_depth': [None], 'min_samples_leaf': [1]}]}

def stripe_image(vertical, size, rng, colour):
    base = np.zeros((size, size, 3), dtype=np.float32)
    index = np.arange(size) // 8 % 2 * 255
    base[:] = index[None, :, None] if vertical else index[:, None, None]
    image = 0.4 * base + 0.6 * np.array(colour, dtype=np.float32) + rng.normal(0, 8, base.shape)
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))

def make_dataset(root, same_colour=False, corrupt_test=False, per_group=3, size=80):
    dataset = Path(root) / 'dataset'
    rng = np.random.default_rng(0)
    images, locations = ([], [])
    for class_name, prefix, label in (('inside', 'in', '1'), ('outside', 'out', '0')):
        for number, split in SPLITS_BY_INDEX.items():
            group = f'{prefix}_{number:03d}'
            folder = dataset / 'images' / class_name / group
            folder.mkdir(parents=True)
            locations.append([group, class_name, f'place {group}', 'yes'])
            for index in range(per_group):
                relative = f'images/{class_name}/{group}/{index}.png'
                if split == 'test' and corrupt_test:
                    (dataset / relative).write_bytes(b'this is not an image')
                else:
                    vertical = class_name == 'inside'
                    colour = (128, 128, 128) if same_colour else (220, 70, 50) if vertical else (40, 80, 210)
                    stripe_image(vertical, size, rng, colour).save(dataset / relative)
                images.append([f'{group}_{index}', relative, label, group, split])
    write_csv(dataset / 'images.csv', ['image_id', 'relative_path', 'label', 'group_id', 'split'], images)
    write_csv(dataset / 'locations.csv', ['group_id', 'class_name', 'location_description', 'boundary_verified'], locations)
    return dataset

def write_csv(path, headers, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.reader(handle))

def edit_images(dataset, edit):
    rows = read_csv(dataset / 'images.csv')
    edit(rows[1:])
    write_csv(dataset / 'images.csv', rows[0], rows[1:])

def edit_locations(dataset, edit):
    rows = read_csv(dataset / 'locations.csv')
    rows = [rows[0]] + edit(rows[1:])
    write_csv(dataset / 'locations.csv', rows[0], rows[1:])
