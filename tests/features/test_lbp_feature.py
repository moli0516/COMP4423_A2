"""Migrated from test_features-2.py: LbpFeatureTests."""

from tests.support.features_fixtures import *

class LbpFeatureTests(unittest.TestCase):

    def test_histogram_length_depends_on_points(self):
        for points, radius in ((8, 1), (16, 2), (24, 3)):
            with self.subTest(points=points):
                config = FeatureConfig(use_lbp=True, lbp_points=points, lbp_radius=radius)
                self.assertEqual(features_api.lbp_features(random_array(), config).shape, (points + 2,))

    def test_histogram_is_normalised_non_negative_float32(self):
        vector = features_api.lbp_features(random_array(4), FeatureConfig(use_lbp=True))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(vector.sum()), 1.0, places=5)
        self.assertTrue(np.all(vector >= 0))

    def test_constant_image_is_dominated_by_flat_pattern(self):
        config = FeatureConfig(use_lbp=True)
        vector = features_api.lbp_features(solid_array((120, 120, 120)), config)
        self.assertEqual(int(np.argmax(vector)), config.lbp_points)
        self.assertGreater(float(vector.max()), 0.9)

    def test_texture_changes_histogram(self):
        config = FeatureConfig(use_lbp=True)
        smooth = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None], (64, 1, 3))
        noisy = random_array(5)
        self.assertGreater(float(np.abs(features_api.lbp_features(smooth, config) - features_api.lbp_features(noisy, config)).sum()), 0.5)
