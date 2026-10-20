"""Migrated from test_train.py: HelperTests."""

from tests.support.training_fixtures import *

class HelperTests(unittest.TestCase):

    def test_expand_grid_is_ordered_and_covers_all_specs(self):
        specs = [{'kernel': ['rbf'], 'C': [1, 10]}, {'kernel': ['linear'], 'C': [0.1]}]
        self.assertEqual(list(expand_grid(specs)), [{'kernel': 'rbf', 'C': 1}, {'kernel': 'rbf', 'C': 10}, {'kernel': 'linear', 'C': 0.1}])

    def test_default_grids_have_expected_trial_counts(self):
        self.assertEqual(len(list(expand_grid(DEFAULT_GRIDS['svm']))), 6)
        self.assertEqual(len(list(expand_grid(DEFAULT_GRIDS['random_forest']))), 4)

    def test_score_predictions_known_values_and_json_serialisable(self):
        scores = score_predictions(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertAlmostEqual(scores['accuracy'], 0.75)
        self.assertAlmostEqual(scores['balanced_accuracy'], 0.75)
        self.assertEqual(scores['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertAlmostEqual(scores['per_class']['inside']['precision'], 2 / 3)
        self.assertEqual(scores['per_class']['inside']['recall'], 1.0)
        self.assertEqual(scores['per_class']['outside']['support'], 2)
        json.dumps(scores)

    def test_score_handles_a_class_that_is_never_predicted(self):
        scores = score_predictions(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]))
        self.assertEqual(scores['per_class']['outside']['precision'], 0.0)
        self.assertEqual(scores['per_class']['outside']['f1'], 0.0)
        self.assertEqual(scores['balanced_accuracy'], 0.5)

    def test_selection_key_orders_by_macro_f1_then_balance_then_accuracy(self):
        low = {'macro_f1': 0.7, 'balanced_accuracy': 0.9, 'accuracy': 0.9}
        high = {'macro_f1': 0.8, 'balanced_accuracy': 0.5, 'accuracy': 0.5}
        tie_better = {'macro_f1': 0.8, 'balanced_accuracy': 0.6, 'accuracy': 0.1}
        self.assertGreater(selection_key(high), selection_key(low))
        self.assertGreater(selection_key(tie_better), selection_key(high))

    def test_to_python_converts_numpy_containers(self):
        converted = to_python({'a': np.array([[1, 2]]), 'b': (np.int64(3), np.float32(1.5))})
        self.assertEqual(converted, {'a': [[1, 2]], 'b': [3, 1.5]})
        json.dumps(converted)

    def test_make_classifier_class_weights_and_unknown_name(self):
        self.assertEqual(make_classifier('svm', {'kernel': 'linear'}, 1, True, 1).class_weight, 'balanced')
        self.assertIsNone(make_classifier('svm', {'kernel': 'linear'}, 1, False, 1).class_weight)
        self.assertEqual(make_classifier('random_forest', {'n_estimators': 5}, 1, True, 1).class_weight, 'balanced_subsample')
        with self.assertRaises(TrainingError):
            make_classifier('neural_network', {}, 1, True, 1)

    def test_feature_columns_match_the_requested_feature_set(self):
        pre = PreprocessingConfig(64)
        all_config = FeatureConfig(use_lbp=True, hog_pixels_per_cell=16)
        np.testing.assert_array_equal(feature_columns(all_config, pre, FeatureConfig(use_hog=False, hog_pixels_per_cell=16)), np.arange(48))
        np.testing.assert_array_equal(feature_columns(all_config, pre, FeatureConfig(use_color=False, hog_pixels_per_cell=16)), np.arange(48, 48 + 324))
        np.testing.assert_array_equal(feature_columns(all_config, pre, FeatureConfig(use_color=False, use_hog=False, use_lbp=True)), np.arange(372, 382))

    def test_path_resolution_rejects_escape_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp).resolve()
            for value in ('../outside.png', '/etc/passwd', 'images/../../x.png'):
                with self.subTest(value=value), self.assertRaises(TrainingError):
                    resolve_image_path(dataset, value)
