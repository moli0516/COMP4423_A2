"""Migrated from test_train.py: SafetyTests."""

from tests.support.training_fixtures import *
from tests.support.training_test_case import TrainingTestCase

class SafetyTests(TrainingTestCase):

    def assert_rejected(self, dataset, message, **overrides):
        with self.assertRaisesRegex(TrainingError, message):
            self.run_training(dataset, **overrides)
        self.assertFalse((self.root / 'run' / 'models' / 'pipeline.joblib').exists())

    def test_requires_a_split_for_every_image(self):
        dataset = make_dataset(self.root)
        edit_images(dataset, lambda rows: rows[0].__setitem__(4, ''))
        self.assert_rejected(dataset, 'split_dataset.py')

    def test_invalid_split_label_and_duplicate_paths_are_rejected(self):
        for name, edit, message in (('split', lambda rows: rows[0].__setitem__(4, 'dev'), 'split must be'), ('label', lambda rows: rows[0].__setitem__(2, '2'), 'label must be'), ('duplicate', lambda rows: rows[1].__setitem__(1, rows[0][1]), 'duplicate relative_path')):
            with self.subTest(case=name):
                dataset = make_dataset(self.root / name)
                edit_images(dataset, edit)
                self.assert_rejected(dataset, message)

    def test_group_cannot_cross_splits_or_labels(self):
        dataset = make_dataset(self.root / 'split')
        edit_images(dataset, lambda rows: rows[0].__setitem__(4, 'test'))
        self.assert_rejected(dataset, 'more than one split')
        dataset = make_dataset(self.root / 'label')
        edit_images(dataset, lambda rows: rows[0].__setitem__(2, '0'))
        self.assert_rejected(dataset, 'both labels')

    def test_each_split_must_contain_both_classes(self):
        dataset = make_dataset(self.root)
        edit_images(dataset, lambda rows: [row.__setitem__(4, 'train') for row in rows if row[3] == 'out_003'])
        self.assert_rejected(dataset, 'validation must contain both classes')

    def test_unverified_groups_are_rejected_by_default(self):
        dataset = make_dataset(self.root)
        edit_locations(dataset, lambda rows: [[*row[:3], 'pending'] if row[0] == 'in_001' else row for row in rows])
        self.assert_rejected(dataset, 'not boundary-verified')

    def test_location_record_must_exist_and_match_class(self):
        dataset = make_dataset(self.root / 'missing')
        edit_locations(dataset, lambda rows: [row for row in rows if row[0] != 'in_001'])
        self.assert_rejected(dataset, 'missing from locations.csv')
        dataset = make_dataset(self.root / 'mismatch')
        edit_locations(dataset, lambda rows: [[row[0], 'outside', *row[2:]] if row[0] == 'in_001' else row for row in rows])
        self.assert_rejected(dataset, 'class conflicts')

    def test_missing_or_escaping_image_paths_are_rejected(self):
        dataset = make_dataset(self.root / 'missing')
        (dataset / 'images/inside/in_001/0.png').unlink()
        self.assert_rejected(dataset, 'file does not exist')
        dataset = make_dataset(self.root / 'escape')
        edit_images(dataset, lambda rows: rows[0].__setitem__(1, '../outside.png'))
        self.assert_rejected(dataset, 'leaves the dataset directory')

    def test_corrupt_training_image_reports_its_path(self):
        dataset = make_dataset(self.root)
        (dataset / 'images/inside/in_001/0.png').write_bytes(b'broken')
        self.assert_rejected(dataset, 'images/inside/in_001/0\\.png')

    def test_unknown_or_empty_options_are_rejected(self):
        dataset = make_dataset(self.root)
        for overrides in ({'feature_sets': ('surf',)}, {'classifiers': ('cnn',)}, {'feature_sets': ()}, {'classifiers': ()}):
            with self.subTest(overrides=overrides), self.assertRaises(TrainingError):
                self.run_training(dataset, **overrides)

    def test_missing_files_and_columns_are_rejected(self):
        with self.assertRaisesRegex(TrainingError, 'Missing file'):
            load_dataset(self.root)
        dataset = make_dataset(self.root / 'columns')
        write_csv(dataset / 'images.csv', ['image_id', 'relative_path'], [['a', 'b']])
        with self.assertRaisesRegex(TrainingError, 'missing columns'):
            load_dataset(dataset)
