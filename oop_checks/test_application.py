import contextlib
import io
import unittest
from unittest.mock import patch

from src.application import Application


class ApplicationTests(unittest.TestCase):
    def test_failed_tests_prevent_training(self):
        app = Application()
        with patch.object(app, "run_tests", return_value=1), patch.object(app, "run_train") as train:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(app.run(["all"]), 1)
            train.assert_not_called()

    def test_successful_tests_forward_arguments(self):
        app = Application()
        with patch.object(app, "run_tests", return_value=0), patch.object(app, "run_train", return_value=0) as train:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run(["all", "--dataset-dir", "dataset_dev"]), 0)
            train.assert_called_once_with(["--dataset-dir", "dataset_dev"])

    def test_help_does_not_run_tests(self):
        app = Application()
        with patch.object(app, "run_tests") as tests, patch.object(app, "run_train", return_value=0) as train:
            self.assertEqual(app.run(["all", "--help"]), 0)
            tests.assert_not_called()
            train.assert_called_once_with(["--help"])

    def test_subprocess_uses_configured_root_and_interpreter(self):
        app = Application(python_executable="custom-python")
        with patch("src.application.application.subprocess.run") as run:
            run.return_value.returncode = 3
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run_module(["src.train"]), 3)
            run.assert_called_once_with(["custom-python", "-m", "src.train"], cwd=app.root, check=False)

    def test_default_root(self):
        from pathlib import Path
        expected = Path(__file__).resolve().parents[1]
        self.assertEqual(Application().root, expected)
