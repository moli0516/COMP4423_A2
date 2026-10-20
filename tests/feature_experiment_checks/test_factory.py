import tempfile
import unittest

import numpy as np

from tests.feature_experiment_checks.helpers import fixture, dimension
from src.experiments.feature_ablation.feature_pipeline_factory import FeaturePipelineFactory
from src.experiments.feature_ablation.feature_variant import FeatureVariant


class FactoryTests(unittest.TestCase):
    def test_four_variants_and_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            _, reference, _ = fixture(folder)
            factory = FeaturePipelineFactory(reference, dimension)
            variants = FeatureVariant.planned_variants()
            self.assertEqual([variant.variant_id for variant in variants], ["F0", "F1", "F2", "F3"])
            self.assertEqual([factory.describe(variant)["feature_dimension"] for variant in variants], [1764, 1812, 1956, 1774])

    def test_classifiers_and_scalers_are_unfitted_independent_clones(self):
        with tempfile.TemporaryDirectory() as folder:
            _, reference, _ = fixture(folder)
            before = reference.named_steps["classifier"].coef_.copy()
            factory = FeaturePipelineFactory(reference, dimension)
            for variant in FeatureVariant.planned_variants():
                candidate = factory.create(variant)
                self.assertFalse(hasattr(candidate.named_steps["classifier"], "coef_"))
                self.assertFalse(hasattr(candidate.named_steps["scale"], "mean_"))
                self.assertIsNot(candidate.named_steps["features"], reference.named_steps["features"])
                self.assertEqual(candidate.named_steps["classifier"].get_params(), reference.named_steps["classifier"].get_params())
                self.assertEqual(candidate.named_steps["features"].preprocessing_config, reference.named_steps["features"].preprocessing_config)
            np.testing.assert_array_equal(reference.named_steps["classifier"].coef_, before)

    def test_wrong_reference_classifier_setting_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, reference, _ = fixture(folder)
            reference.named_steps["classifier"].set_params(C=1.0)
            with self.assertRaisesRegex(ValueError, "C=0.01"):
                FeaturePipelineFactory(reference, dimension)

    def test_only_intended_feature_flags_change(self):
        with tempfile.TemporaryDirectory() as folder:
            _, reference, _ = fixture(folder)
            base = reference.named_steps["features"].feature_config
            factory = FeaturePipelineFactory(reference, dimension)
            for variant in FeatureVariant.planned_variants():
                feature_config = factory.create(variant).named_steps["features"].feature_config
                self.assertEqual(feature_config.hog_pixels_per_cell, base.hog_pixels_per_cell)
                self.assertEqual(feature_config.color_bins, base.color_bins)
                self.assertEqual(feature_config.lbp_points, base.lbp_points)
                self.assertTrue(feature_config.use_hog)
