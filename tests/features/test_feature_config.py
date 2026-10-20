"""Migrated from test_features-2.py: FeatureConfigTests."""

from tests.support.features_fixtures import *

class FeatureConfigTests(unittest.TestCase):

    def test_defaults_and_optional_lbp(self):
        config = FeatureConfig()
        self.assertTrue(config.use_color and config.use_hog)
        self.assertFalse(config.use_lbp)
        self.assertEqual(features_public.DEFAULT_FEATURE_CONFIG, config)

    def test_at_least_one_feature_type_is_required(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            FeatureConfig(use_color=False, use_hog=False, use_lbp=False)

    def test_single_feature_types_are_allowed(self):
        for values in ({'use_hog': False}, {'use_color': False}, {'use_color': False, 'use_hog': False, 'use_lbp': True}):
            with self.subTest(values=values):
                self.assertIsInstance(FeatureConfig(**values), FeatureConfig)

    def test_invalid_numbers_are_rejected(self):
        names = ['color_bins', 'color_grid', 'hog_orientations', 'hog_pixels_per_cell', 'hog_cells_per_block', 'lbp_points', 'lbp_radius']
        for name in names:
            for value in (0, -1, 1.5, '8', None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    FeatureConfig(**{name: value})
        with self.assertRaises(ValueError):
            FeatureConfig(color_bins=1)

    def test_config_is_immutable_and_hashable(self):
        config = FeatureConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.color_bins = 4
        self.assertEqual(hash(config), hash(FeatureConfig()))

    def test_dict_round_trip_and_unknown_keys(self):
        config = FeatureConfig(color_bins=8, color_grid=2, use_lbp=True, lbp_points=16)
        self.assertEqual(FeatureConfig.from_dict(config.to_dict()), config)
        self.assertEqual(FeatureConfig.from_dict({'color_bins': 4}).color_bins, 4)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            FeatureConfig.from_dict({'surf': True})
