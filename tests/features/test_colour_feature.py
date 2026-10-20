"""Migrated from test_features-2.py: ColourFeatureTests."""

from tests.support.features_fixtures import *

class ColourFeatureTests(unittest.TestCase):

    def test_primary_colour_histogram_positions(self):
        cases = {'red': ((255, 0, 0), (0, 15, 15)), 'blue': ((0, 0, 255), (10, 15, 15)), 'black': ((0, 0, 0), (0, 0, 0)), 'white': ((255, 255, 255), (0, 0, 15))}
        for name, (colour, bins) in cases.items():
            with self.subTest(colour=name):
                vector = features_api.color_histogram_features(solid_array(colour), FeatureConfig())
                expected = np.zeros(48, dtype=np.float32)
                for channel, bin_index in enumerate(bins):
                    expected[channel * 16 + bin_index] = 1
                np.testing.assert_allclose(vector, expected)

    def test_dimensions_follow_bins_and_grid(self):
        image = random_array()
        for bins in (2, 8, 16):
            for grid in (1, 2, 3):
                with self.subTest(bins=bins, grid=grid):
                    vector = features_api.color_histogram_features(image, FeatureConfig(color_bins=bins, color_grid=grid))
                    self.assertEqual(vector.shape, (3 * bins * grid * grid,))
                    self.assertEqual(vector.dtype, np.float32)

    def test_each_channel_in_each_grid_cell_sums_to_one(self):
        image = random_array(size=70)
        vector = features_api.color_histogram_features(image, FeatureConfig(color_bins=8, color_grid=3))
        np.testing.assert_allclose(vector.reshape(27, 8).sum(axis=1), np.ones(27), atol=1e-06)
        self.assertTrue(np.all((vector >= 0) & (vector <= 1)))

    def test_grid_keeps_coarse_spatial_layout(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :32, 0] = 255
        image[:, 32:, 2] = 255
        vector = features_api.color_histogram_features(image, FeatureConfig(color_grid=2))
        top_left_hue, top_right_hue = (vector[:16], vector[48:64])
        self.assertEqual(int(np.argmax(top_left_hue)), 0)
        self.assertEqual(int(np.argmax(top_right_hue)), 10)
        global_vector = features_api.color_histogram_features(image, FeatureConfig(color_grid=1))
        self.assertAlmostEqual(float(global_vector[0]), 0.5)
        self.assertAlmostEqual(float(global_vector[10]), 0.5)

    def test_empty_grid_cells_do_not_create_invalid_values(self):
        vector = features_api.color_histogram_features(solid_array((1, 2, 3), size=16), FeatureConfig(color_grid=20))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertEqual(vector.shape, (3 * 16 * 400,))

    def test_hue_histogram_is_unchanged_by_brightness_scaling(self):
        bright = features_api.color_histogram_features(solid_array((200, 100, 50)), FeatureConfig())
        dark = features_api.color_histogram_features(solid_array((100, 50, 25)), FeatureConfig())
        np.testing.assert_array_equal(bright[:16], dark[:16])
        self.assertFalse(np.array_equal(bright[32:], dark[32:]))
