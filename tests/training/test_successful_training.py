"""Migrated from test_train.py: SuccessfulTrainingTests."""

from tests.support.training_fixtures import *
from tests.support.training_test_case import TrainingTestCase

class SuccessfulTrainingTests(TrainingTestCase):

    def test_outputs_pipeline_metadata_trials_and_report(self):
        dataset = make_dataset(self.root)
        result = self.run_training(dataset)
        models, results = (result['models_dir'], result['results_dir'])
        for path in (models / 'pipeline.joblib', models / 'pipeline_metadata.json', results / 'validation_experiments.csv', results / 'training_report.json'):
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path.name)
        trials = read_csv(results / 'validation_experiments.csv')
        self.assertEqual(len(trials) - 1, 4)
        self.assertIn('validation_macro_f1', trials[0])
        metadata = json.loads((models / 'pipeline_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['class_labels'], {'0': 'outside', '1': 'inside'})
        self.assertEqual(metadata['trained_on_splits'], ['train'])
        self.assertEqual(metadata['selected_using_splits'], ['validation'])
        self.assertEqual(metadata['test_images_opened_by_training'], 0)
        self.assertEqual(metadata['images_csv_sha256'], hashlib.sha256((dataset / 'images.csv').read_bytes()).hexdigest())
        self.assertEqual(PreprocessingConfig.from_dict(metadata['preprocessing_config']), PreprocessingConfig(64))
        self.assertEqual(metadata['counts']['test']['inside'], {'images': 3, 'groups': 1})

    def test_test_images_are_never_opened(self):
        dataset = make_dataset(self.root, corrupt_test=True)
        result = self.run_training(dataset)
        self.assertEqual(result['report']['test_images_opened_by_training'], 0)

    def test_loaded_pipeline_predicts_validation_images_from_paths_and_bytes(self):
        dataset = make_dataset(self.root)
        result = self.run_training(dataset)
        pipeline = joblib.load(result['models_dir'] / 'pipeline.joblib')
        paths = [dataset / 'images/inside/in_003/0.png', dataset / 'images/outside/out_003/0.png']
        self.assertEqual(pipeline.predict(paths).tolist(), [1, 0])
        self.assertEqual(pipeline.predict([paths[0].read_bytes(), paths[1].read_bytes()]).tolist(), [1, 0])
        self.assertEqual(list(pipeline.named_steps), ['features', 'classifier'] if result['report']['classifier'] == 'random_forest' else ['features', 'scale', 'classifier'])

    def test_ties_are_resolved_by_earliest_trial(self):
        dataset = make_dataset(self.root)
        result = self.run_training(dataset)
        trials = result['trials']
        self.assertTrue(all((row['validation_macro_f1'] == 1.0 for row in trials)))
        self.assertEqual(result['report']['best_trial']['trial'], 1)

    def test_validation_selects_the_feature_set_that_actually_works(self):
        dataset = make_dataset(self.root, same_colour=True)
        result = self.run_training(dataset, feature_sets=('color', 'hog'), classifiers=('svm',))
        by_set = {row['feature_set']: row['validation_macro_f1'] for row in result['trials']}
        self.assertLess(by_set['color'], by_set['hog'])
        self.assertEqual(result['report']['feature_set'], 'hog')
        self.assertEqual(result['report']['validation_metrics']['macro_f1'], 1.0)
        self.assertEqual(result['report']['best_trial']['validation_macro_f1'], max(by_set.values()))

    def test_training_is_reproducible_for_the_same_seed(self):
        dataset = make_dataset(self.root)
        first = self.run_training(dataset, 'first', seed=5)
        second = self.run_training(dataset, 'second', seed=5)
        drop_time = lambda rows: [{key: value for key, value in row.items() if key != 'fit_seconds'} for row in rows]
        self.assertEqual(drop_time(first['trials']), drop_time(second['trials']))
        paths = [dataset / 'images/inside/in_003/1.png', dataset / 'images/outside/out_003/1.png']
        np.testing.assert_array_equal(first['pipeline'].predict(paths), second['pipeline'].predict(paths))

    def test_class_weight_option_is_recorded_and_applied(self):
        dataset = make_dataset(self.root)
        balanced = self.run_training(dataset, 'balanced', classifiers=('svm',), feature_sets=('color',))
        plain = self.run_training(dataset, 'plain', classifiers=('svm',), feature_sets=('color',), balanced=False)
        self.assertEqual(balanced['pipeline'].named_steps['classifier'].class_weight, 'balanced')
        self.assertIsNone(plain['pipeline'].named_steps['classifier'].class_weight)
        self.assertIsNone(plain['report']['class_weight'])

    def test_development_flag_is_recorded_when_unverified_data_is_allowed(self):
        dataset = make_dataset(self.root)
        edit_locations(dataset, lambda rows: [[*row[:3], 'pending'] for row in rows])
        result = self.run_training(dataset, allow_unverified=True)
        self.assertTrue(result['report']['development_run'])

    def test_serialisation_failure_leaves_no_partial_outputs(self):
        dataset = make_dataset(self.root)
        models, results = (self.root / 'failed' / 'models', self.root / 'failed' / 'results')
        with mock.patch.object(artifact_store_module.json, 'dumps', side_effect=TypeError('cannot serialise')):
            with self.assertRaises(TypeError):
                run_training(dataset, models, results, ('color',), ('svm',), 64, hog_cell=16, grids=FAST_GRIDS, log=lambda *_: None)
        self.assertEqual(list(models.glob('*')), [])
        self.assertEqual(list(results.glob('*')), [])
