import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from polyu_lr_checks.helpers import fixture
from src.experiments.polyu_lr_config import PolyuLRConfig


class ConfigTests(unittest.TestCase):
    def test_json_roundtrip_with_root_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = {"dataset_dir": "dataset", "reference_metadata": "meta.json", "reference_report": "report.json", "output_dir": "output"}
            path = root / "config.json"
            path.write_text(json.dumps(values))
            config = PolyuLRConfig.from_json(path, root)
            self.assertEqual(config.dataset_dir, (root / "dataset").resolve())
            self.assertEqual(config.c_values, (0.01, 0.1, 1.0))

    def test_scope_and_type_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            for changes in ({"solver": "saga"}, {"c_values": (0.1, 1, 10)}, {"c_values": (0.01, 0.1, True)}, {"seed": True}, {"max_iter": 10000.0}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replace(config, **changes)

    def test_immutable_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            with self.assertRaises(FrozenInstanceError):
                config.seed = 10

    def test_unknown_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"unexpected": True}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                PolyuLRConfig.from_json(path, folder)
