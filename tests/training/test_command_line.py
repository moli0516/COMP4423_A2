"""Migrated from test_train.py: CommandLineTests."""

from tests.support.training_fixtures import *
from tests.support.training_test_case import TrainingTestCase

class CommandLineTests(TrainingTestCase):

    def run_main(self, arguments):
        stdout, stderr = (io.StringIO(), io.StringIO())
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = training_cli.main(arguments)
        return (code, stdout.getvalue(), stderr.getvalue())

    def arguments(self, dataset, name='cli'):
        return ['--dataset-dir', str(dataset), '--models-dir', str(self.root / name / 'models'), '--results-dir', str(self.root / name / 'results'), '--image-size', '64', '--hog-cell', '16', '--feature-sets', 'color', '--classifiers', 'svm']

    def test_main_runs_end_to_end_and_warns_not_to_touch_test_split(self):
        dataset = make_dataset(self.root)
        code, out, err = self.run_main(self.arguments(dataset))
        self.assertEqual(code, 0, err)
        self.assertIn('Selected: color + svm', out)
        self.assertIn('test split was not opened', out)
        self.assertTrue((self.root / 'cli' / 'models' / 'pipeline.joblib').exists())

    def test_allow_unverified_requires_explicit_output_directories(self):
        dataset = make_dataset(self.root)
        code, out, err = self.run_main(['--dataset-dir', str(dataset), '--allow-unverified'])
        self.assertEqual(code, 1)
        self.assertIn('--models-dir', err)
        self.assertEqual(out, '')

    def test_development_run_is_labelled_in_output(self):
        dataset = make_dataset(self.root)
        edit_locations(dataset, lambda rows: [[*row[:3], 'pending'] for row in rows])
        code, out, err = self.run_main(self.arguments(dataset) + ['--allow-unverified'])
        self.assertEqual(code, 0, err)
        self.assertIn('DEVELOPMENT RUN', out)

    def test_main_returns_one_with_readable_error(self):
        code, out, err = self.run_main(['--dataset-dir', str(self.root / 'missing')])
        self.assertEqual(code, 1)
        self.assertIn('ERROR: Missing file', err)

    def test_invalid_choice_is_rejected_by_argument_parser(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            training_cli.main(['--classifiers', 'cnn'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('invalid choice', stderr.getvalue())
