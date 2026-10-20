"""Migrated from test_features-2.py: ExtractFeatureTests."""

from tests.support.features_fixtures import *

class ExtractFeatureTests(unittest.TestCase):

    def test_every_block_function_returns_one_dimensional_float32(self):
        image = random_array(13)
        config = FeatureConfig(hog_pixels_per_cell=16, use_lbp=True, color_grid=2)
        for name, function in {'color': features_api.color_histogram_features, 'hog': features_api.hog_features, 'lbp': features_api.lbp_features}.items():
            with self.subTest(block=name):
                vector = function(image, config)
                self.assertEqual(vector.ndim, 1)
                self.assertEqual(vector.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(vector)))

    def test_blocks_are_ordered_and_concatenated_in_that_order(self):
        image = random_array(6)
        blocks = features_api.extract_feature_blocks(image, ALL_BLOCKS)
        self.assertEqual(list(blocks), ['color', 'hog', 'lbp'])
        np.testing.assert_array_equal(features_api.extract_features(image, ALL_BLOCKS), np.concatenate(list(blocks.values())))

    def test_enabled_blocks_match_configuration(self):
        image = random_array(7)
        cases = {'color_only': (FeatureConfig(use_hog=False), ['color']), 'hog_only': (FeatureConfig(use_color=False, hog_pixels_per_cell=16), ['hog']), 'lbp_only': (FeatureConfig(use_color=False, use_hog=False, use_lbp=True), ['lbp']), 'all': (ALL_BLOCKS, ['color', 'hog', 'lbp'])}
        for name, (config, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(list(features_api.extract_feature_blocks(image, config)), expected)

    def test_vector_is_finite_float32_and_deterministic(self):
        image = random_array(8)
        first, second = (features_api.extract_features(image, ALL_BLOCKS), features_api.extract_features(image.copy(), ALL_BLOCKS))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.ndim, 1)
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)

    def test_extraction_does_not_mutate_input(self):
        image = random_array(9)
        before = image.copy()
        features_api.extract_features(image, ALL_BLOCKS)
        np.testing.assert_array_equal(image, before)

    def test_invalid_images_are_rejected(self):
        invalid = {'float': random_array().astype(np.float32), 'grayscale': random_array()[:, :, 0], 'rgba': np.zeros((64, 64, 4), dtype=np.uint8), 'tiny': random_array(size=8), 'list': random_array().tolist(), 'none': None, 'batch': np.zeros((2, 64, 64, 3), dtype=np.uint8)}
        for name, image in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                features_api.extract_features(image, SMALL)

    def test_non_square_images_are_supported(self):
        image = random_array(size=96)[:64]
        self.assertEqual(features_api.extract_features(image, SMALL).ndim, 1)

    def test_sizes_match_actual_vectors(self):
        sizes = features_api.feature_block_sizes(ALL_BLOCKS, SMALL_PRE)
        blocks = features_api.extract_feature_blocks(random_array(10), ALL_BLOCKS)
        self.assertEqual(sizes, {name: values.size for name, values in blocks.items()})
        self.assertEqual(features_api.feature_dimension(ALL_BLOCKS, SMALL_PRE), sum(sizes.values()))

    def test_documented_default_dimensions(self):
        self.assertEqual(features_api.feature_block_sizes(), {'color': 48, 'hog': 1764})
        self.assertEqual(features_api.feature_dimension(), 1812)
        self.assertEqual(features_api.feature_dimension(FeatureConfig(use_lbp=True)), 1822)

    def test_dimension_is_independent_of_original_photo_size_after_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [write_image(Path(temp) / f'{index}.png', random_array(index, size=size)[:size - index * 3]) for index, size in enumerate((40, 80, 200))]
            lengths = {features_api.extract_features(preprocessing_api.load_and_preprocess(path, SMALL_PRE), SMALL).size for path in paths}
        self.assertEqual(lengths, {features_api.feature_dimension(SMALL, SMALL_PRE)})
