"""Migrated from test_features-2.py: PipelineIntegrationTests."""

from tests.support.features_fixtures import *

class PipelineIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(123)
        self.paths, self.labels = ([], [])
        for index in range(16):
            label = index % 2
            base = stripes(vertical=bool(label), size=80).astype(np.float32)
            colour = np.array([220, 80, 60] if label else [40, 90, 210], dtype=np.float32)
            image = 0.45 * base + 0.55 * colour + rng.normal(0, 8, base.shape)
            self.paths.append(write_image(self.root / f'sample_{index}.png', image.clip(0, 255).astype(np.uint8)))
            self.labels.append(label)

    def make_pipeline(self):
        return Pipeline([('features', ImageFeatureExtractor(SMALL_PRE, FeatureConfig(hog_pixels_per_cell=16))), ('scale', StandardScaler()), ('classifier', LogisticRegression(max_iter=2000, random_state=0))])

    def test_pipeline_learns_separable_image_classes(self):
        model = self.make_pipeline().fit(self.paths[:12], self.labels[:12])
        self.assertEqual(model.predict(self.paths[12:]).tolist(), self.labels[12:])
        self.assertEqual(model.score(self.paths[:12], self.labels[:12]), 1.0)

    def test_saved_pipeline_reproduces_predictions_for_paths_and_bytes(self):
        model = self.make_pipeline().fit(self.paths[:12], self.labels[:12])
        model_path = self.root / 'pipeline.joblib'
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)
        np.testing.assert_allclose(loaded.decision_function(self.paths[12:]), model.decision_function(self.paths[12:]))
        self.assertEqual(loaded.predict([self.paths[13].read_bytes()]).tolist(), model.predict([self.paths[13]]).tolist())
        self.assertEqual(loaded.named_steps['features'].preprocessing_config, SMALL_PRE)

    def test_pipeline_clone_is_unfitted_but_keeps_feature_settings(self):
        model = self.make_pipeline()
        copied = clone(model)
        self.assertEqual(copied.named_steps['features'].feature_config, model.named_steps['features'].feature_config)
        copied.fit(self.paths[:12], self.labels[:12])
        with self.assertRaises(Exception):
            model.predict(self.paths[12:])
