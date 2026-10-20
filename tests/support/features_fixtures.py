import contextlib
import dataclasses
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import src.features as features_public
import src.preprocessing as preprocessing_public
from src.features import api as features_api
from src.preprocessing import api as preprocessing_api
from src.features.feature_config import FeatureConfig
from src.features.image_feature_extractor import ImageFeatureExtractor
from src.preprocessing.preprocessing_config import PreprocessingConfig

SMALL_PRE = PreprocessingConfig(image_size=64, resize_mode='squash')
SMALL = FeatureConfig(hog_pixels_per_cell=16)
ALL_BLOCKS = FeatureConfig(hog_pixels_per_cell=16, use_lbp=True)

def solid_array(colour, size=64):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = colour
    return image

def random_array(seed=0, size=64):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)

def stripes(vertical, size=64, period=8):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    index = np.arange(size) // period % 2 * 255
    if vertical:
        image[:, :, :] = index[None, :, None]
    else:
        image[:, :, :] = index[:, None, None]
    return image

def write_image(path, array):
    Image.fromarray(array).save(path)
    return path
