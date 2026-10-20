import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.base import clone

from src.preprocessing import ImagePreprocessor, PreprocessingConfig, load_and_preprocess
from src.features import (
    FeatureConfig, VisualFeatureExtractor, ImageFeatureExtractor,
    feature_block_sizes, extract_features,
)
from src.training.constants import FEATURE_SETS


class PreprocessingAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.random.default_rng(42).integers(0, 256, (91, 153, 3), dtype=np.uint8)

    def test_all_resize_modes(self):
        for mode in ("squash", "center_crop", "pad"):
            with self.subTest(mode=mode):
                result = ImagePreprocessor(PreprocessingConfig(256, mode, 17)).transform(self.pixels)
                self.assertEqual(result.shape, (256, 256, 3))
                self.assertEqual(result.dtype, np.uint8)

    def test_pad_background(self):
        result = ImagePreprocessor(PreprocessingConfig(256, "pad", 17)).transform(self.pixels)
        self.assertTrue(np.all(result[0, 0] == 17))

    def test_grayscale_becomes_rgb(self):
        result = ImagePreprocessor().transform(self.pixels[:, :, 0])
        self.assertEqual(result.shape, (256, 256, 3))

    def test_invalid_preprocessing_config(self):
        for options in ({"image_size": 0}, {"resize_mode": "invalid"}, {"pad_value": 256}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PreprocessingConfig(**options)

    def test_float_pixels_are_rejected(self):
        with self.assertRaises(ValueError):
            ImagePreprocessor().transform(self.pixels.astype(float))

    def test_default_feature_dimensions(self):
        image = ImagePreprocessor().transform(self.pixels)
        for name, size in (("color", 48), ("hog", 1764), ("color_hog", 1812), ("color_hog_lbp", 1822)):
            with self.subTest(feature_set=name):
                config = FeatureConfig(**FEATURE_SETS[name])
                vector = VisualFeatureExtractor(config).extract(image)
                self.assertEqual(vector.shape, (size,))
                self.assertEqual(sum(feature_block_sizes(config).values()), size)
                self.assertTrue(np.isfinite(vector).all())
                np.testing.assert_array_equal(vector, extract_features(image, config))

    def test_disabled_features_rejected(self):
        with self.assertRaises(ValueError):
            FeatureConfig(use_color=False, use_hog=False, use_lbp=False)

    def test_transformer_clone_and_path_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "photo.png"
            Image.fromarray(self.pixels).save(path)
            extractor = ImageFeatureExtractor(PreprocessingConfig(), FeatureConfig())
            copied = clone(extractor)
            actual = copied.fit_transform([path])
            expected = extract_features(load_and_preprocess(path))
            np.testing.assert_allclose(actual[0], expected)

    def test_empty_path_sequence_rejected(self):
        with self.assertRaises(ValueError):
            ImageFeatureExtractor().transform([])

    def test_single_string_path_rejected(self):
        with self.assertRaises(ValueError):
            ImageFeatureExtractor().transform("photo.png")
