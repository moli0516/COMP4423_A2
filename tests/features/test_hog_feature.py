"""Migrated from test_features-2.py: HogFeatureTests."""

from tests.support.features_fixtures import *

class HogFeatureTests(unittest.TestCase):

    def test_expected_dimensions(self):
        cases = {(16, 9, 2): 324, (32, 9, 2): 36, (16, 6, 2): 216, (16, 9, 1): 4 * 4 * 1 * 1 * 9}
        for (cell, orientations, block), expected in cases.items():
            with self.subTest(cell=cell, orientations=orientations, block=block):
                config = FeatureConfig(hog_pixels_per_cell=cell, hog_orientations=orientations, hog_cells_per_block=block)
                vector = features_api.hog_features(random_array(), config)
                self.assertEqual(vector.shape, (expected,))
                self.assertEqual(vector.dtype, np.float32)

    def test_default_dimension_for_256_pixel_images(self):
        self.assertEqual(features_api.hog_features(random_array(size=256), FeatureConfig()).shape, (1764,))

    def test_constant_image_has_no_gradient_energy(self):
        np.testing.assert_array_equal(features_api.hog_features(solid_array((120, 120, 120)), SMALL), np.zeros(324, dtype=np.float32))

    def test_orientation_of_edges_changes_descriptor(self):
        vertical = features_api.hog_features(stripes(True), SMALL)
        horizontal = features_api.hog_features(stripes(False), SMALL)
        self.assertGreater(float(np.linalg.norm(vertical)), 0.1)
        self.assertGreater(float(np.linalg.norm(vertical - horizontal)), 0.5)

    def test_hog_uses_luminance_from_all_colour_channels(self):
        for channel in range(3):
            with self.subTest(striped_channel=channel):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:, :, channel] = stripes(True)[:, :, 0]
                self.assertGreater(float(np.linalg.norm(features_api.hog_features(image, SMALL))), 0.1)

    def test_output_is_finite_non_negative_and_deterministic(self):
        image = random_array(3)
        first, second = (features_api.hog_features(image, SMALL), features_api.hog_features(image.copy(), SMALL))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(first >= 0))

    def test_too_small_image_for_hog_fails_clearly(self):
        with self.assertRaises(ValueError):
            features_api.hog_features(random_array(size=31), FeatureConfig())
