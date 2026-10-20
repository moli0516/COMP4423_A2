import tempfile
import unittest
from unittest.mock import patch

from tests.polyu_lr_checks.helpers import fixture
from tests.polyu_lr_checks.hog_test_transformer import HogTestTransformer
from src.experiments.polyu_pipeline_factory import PolyuPipelineFactory


class FactoryTests(unittest.TestCase):
    def test_exact_candidate_count_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            candidates = list(PolyuPipelineFactory(config, "balanced").candidates())
            self.assertEqual(len(candidates), 7)
            self.assertEqual(candidates[0]["family"], "svm")
            self.assertEqual([c["penalty"] for c in candidates[1:]], ["l2"] * 3 + ["l1"] * 3)

    def test_full_pipeline_and_fixed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            factory = PolyuPipelineFactory(config, "balanced")
            transformer = HogTestTransformer()
            for candidate in factory.candidates():
                pipeline = factory.create(candidate, transformer)
                self.assertEqual(list(pipeline.named_steps), ["features", "scale", "classifier"])
                self.assertIsNot(pipeline.named_steps["features"], transformer)
                self.assertEqual(pipeline.named_steps["classifier"].class_weight, "balanced")

    def test_new_api_regularization_routing(self):
        def new_constructor(*, penalty="deprecated", l1_ratio=0.0):
            pass
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            with patch("src.experiments.polyu_pipeline_factory.LogisticRegression", new_constructor):
                factory = PolyuPipelineFactory(config, "balanced")
                for penalty, ratio in (("l1", 1.0), ("l2", 0.0)):
                    parameters = factory.logistic_parameters(penalty, 0.1)
                    self.assertEqual(parameters["l1_ratio"], ratio)
                    self.assertNotIn("penalty", parameters)

    def test_old_api_regularization_routing(self):
        def old_constructor(*, penalty="l2", l1_ratio=None):
            pass
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = fixture(folder)
            with patch("src.experiments.polyu_pipeline_factory.LogisticRegression", old_constructor):
                parameters = PolyuPipelineFactory(config, None).logistic_parameters("l1", 0.1)
                self.assertEqual(parameters["penalty"], "l1")
                self.assertNotIn("l1_ratio", parameters)
