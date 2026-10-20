"""Migrated from test_features-2.py: ImageFeatureExtractorTests."""

from tests.support.features_fixtures import *

class ImageFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [write_image(self.root / f'image_{index}.png', random_array(index, 90 + index)) for index in range(4)]
        self.extractor = ImageFeatureExtractor(SMALL_PRE, ALL_BLOCKS)

    def test_fit_is_stateless_and_returns_self(self):
        before = dataclasses.replace(self.extractor.feature_config)
        self.assertIs(self.extractor.fit(self.paths, [0, 1, 0, 1]), self.extractor)
        self.assertEqual(self.extractor.feature_config, before)

    def test_transform_shape_dtype_and_finite_values(self):
        matrix = self.extractor.transform(self.paths)
        self.assertEqual(matrix.shape, (4, features_api.feature_dimension(ALL_BLOCKS, SMALL_PRE)))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(matrix)))

    def test_rows_match_manual_pipeline_and_preserve_order(self):
        matrix = self.extractor.transform(self.paths)
        for index, path in enumerate(self.paths):
            manual = features_api.extract_features(preprocessing_api.load_and_preprocess(path, SMALL_PRE), ALL_BLOCKS)
            np.testing.assert_array_equal(matrix[index], manual)
        np.testing.assert_array_equal(self.extractor.transform(self.paths[::-1]), matrix[::-1])

    def test_empty_input_has_correct_column_count(self):
        empty = self.extractor.transform([])
        self.assertEqual(empty.shape, (0, features_api.feature_dimension(ALL_BLOCKS, SMALL_PRE)))
        self.assertEqual(empty.dtype, np.float32)

    def test_accepts_paths_bytes_streams_pil_and_generators(self):
        path = self.paths[0]
        data = path.read_bytes()
        reference = self.extractor.transform([path])[0]
        sources = [str(path), data, io.BytesIO(data), Image.open(path)]
        for source in sources:
            with self.subTest(source=type(source).__name__):
                np.testing.assert_array_equal(self.extractor.transform([source])[0], reference)
        np.testing.assert_array_equal(self.extractor.transform((item for item in [path])), reference[None, :])

    def test_bad_input_raises_preprocessing_error(self):
        bad = self.root / 'bad.png'
        bad.write_bytes(b'bad')
        for source in (bad, self.root / 'missing.png'):
            with self.subTest(source=source.name), self.assertRaises(preprocessing_public.PreprocessingError):
                self.extractor.transform([self.paths[0], source])

    def test_file_names_and_folders_do_not_change_features(self):
        first = self.root / 'inside' / 'in_001' / 'campus.png'
        second = self.root / 'outside' / 'out_001' / 'other_name.png'
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_bytes(self.paths[0].read_bytes())
        second.write_bytes(self.paths[0].read_bytes())
        np.testing.assert_array_equal(self.extractor.transform([first]), self.extractor.transform([second]))

    def test_estimator_parameter_api_and_clone(self):
        parameters = self.extractor.get_params()
        self.assertEqual(parameters['preprocessing_config'], SMALL_PRE)
        self.assertEqual(parameters['feature_config'], ALL_BLOCKS)
        copied = clone(self.extractor)
        self.assertEqual(copied.get_params(), parameters)
        copied.set_params(feature_config=SMALL)
        self.assertEqual(self.extractor.feature_config, ALL_BLOCKS)
        self.assertLess(copied.transform(self.paths[:1]).shape[1], self.extractor.transform(self.paths[:1]).shape[1])

    def test_preprocessing_config_changes_feature_values_or_dimensions(self):
        wide = write_image(self.root / 'wide.png', random_array(12, 120)[:60])
        squash = ImageFeatureExtractor(SMALL_PRE, SMALL).transform([wide])
        padded = ImageFeatureExtractor(PreprocessingConfig(64, 'pad', 0), SMALL).transform([wide])
        larger = ImageFeatureExtractor(PreprocessingConfig(96, 'squash'), SMALL).transform([wide])
        self.assertFalse(np.array_equal(squash, padded))
        self.assertNotEqual(squash.shape, larger.shape)
