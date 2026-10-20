"""Unit tests for the Logistic Regression (L1/L2) classifier option."""
import unittest

import numpy as np

from src.exceptions import TrainingError
from src.training.classifier_factory import make_classifier
from src.training.constants import CLASSIFIERS, DEFAULT_GRIDS, SCALED_CLASSIFIERS
from src.training.search import expand_grid


def sample_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 40))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=120) > 0).astype(int)
    return x, y


class LogisticRegressionOptionTests(unittest.TestCase):
    def test_registered_scaled_and_grid_size(self):
        self.assertIn("logistic_regression", CLASSIFIERS)
        self.assertIn("logistic_regression", SCALED_CLASSIFIERS)
        self.assertEqual(len(list(expand_grid(DEFAULT_GRIDS["logistic_regression"]))), 6)
        self.assertEqual({p["penalty"] for p in expand_grid(DEFAULT_GRIDS["logistic_regression"])}, {"l1", "l2"})

    def test_l1_is_sparser_than_l2(self):
        x, y = sample_data()
        l1 = make_classifier("logistic_regression", {"penalty": "l1", "C": 0.1}, 0, True, 1).fit(x, y)
        l2 = make_classifier("logistic_regression", {"penalty": "l2", "C": 0.1}, 0, True, 1).fit(x, y)
        self.assertLess(np.count_nonzero(l1.coef_), np.count_nonzero(l2.coef_))
        self.assertEqual(np.count_nonzero(l2.coef_), 40)

    def test_class_weight_and_scores(self):
        x, y = sample_data()
        model = make_classifier("logistic_regression", {"penalty": "l1", "C": 1.0}, 0, True, 1).fit(x, y)
        self.assertEqual(model.class_weight, "balanced")
        self.assertIsNone(make_classifier("logistic_regression", {"C": 1.0}, 0, False, 1).class_weight)
        self.assertEqual(model.decision_function(x).shape, (120,))

    def test_default_penalty_is_l2_and_bad_penalty_is_rejected(self):
        x, y = sample_data()
        self.assertEqual(np.count_nonzero(make_classifier("logistic_regression", {"C": 0.1}, 0, True, 1).fit(x, y).coef_), 40)
        with self.assertRaises(TrainingError):
            make_classifier("logistic_regression", {"penalty": "elasticnet"}, 0, True, 1)

    def test_other_classifiers_still_unknown_names_rejected(self):
        with self.assertRaises(TrainingError):
            make_classifier("neural_network", {}, 0, True, 1)


if __name__ == "__main__":
    unittest.main()
