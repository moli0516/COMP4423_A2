import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.feature_ablation_checks.helpers import build_case

ROOT = Path(__file__).resolve().parents[2]


def run_script(*arguments):
    return subprocess.run([sys.executable, "scripts/run_feature_experiment.py", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=25)


class CliTests(unittest.TestCase):
    def test_help_lists_gate_and_dry_run(self):
        result = run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--confirm-camera-diagnostic-completed", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_dry_run_gate_block_and_confirmed_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            case = build_case(folder)
            output = case["root"] / "cli_run"
            dry = run_script("--config", str(case["config_path"]), "--output-dir", str(output), "--dry-run")
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn("DRY RUN", dry.stdout)
            self.assertFalse(output.exists())
            blocked = run_script("--config", str(case["config_path"]), "--output-dir", str(output))
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("camera_diagnostic_status is pending", blocked.stderr)
            self.assertFalse(output.exists())
            completed = run_script("--config", str(case["config_path"]), "--output-dir", str(output), "--confirm-camera-diagnostic-completed")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "best/pipeline.joblib").is_file())
            self.assertIn("Best model", completed.stdout)
