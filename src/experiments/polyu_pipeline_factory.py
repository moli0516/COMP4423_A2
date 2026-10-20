from inspect import signature

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class PolyuPipelineFactory:
    """Construct unfitted full-image pipelines with explicit version routing."""

    def __init__(self, config, class_weight):
        self.config = config
        self.class_weight = class_weight

    def logistic_parameters(self, penalty, c_value):
        if penalty not in {"l1", "l2"}:
            raise ValueError("Only L1/L2 are part of this experiment.")
        values = {
            "C": c_value, "solver": self.config.solver,
            "max_iter": self.config.max_iter, "tol": self.config.tol,
            "random_state": self.config.seed, "class_weight": self.class_weight,
        }
        parameters = signature(LogisticRegression).parameters
        penalty_parameter = parameters.get("penalty")
        new_api = penalty_parameter is None or penalty_parameter.default == "deprecated"
        if new_api:
            if "l1_ratio" not in parameters:
                raise RuntimeError("Unsupported LogisticRegression regularization API.")
            values["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
        else:
            values["penalty"] = penalty
        return values

    def candidates(self):
        yield {"id": "baseline_hog_svm_linear_c_0_1", "family": "svm", "penalty": None, "C": 0.1}
        for penalty in ("l2", "l1"):
            for c_value in self.config.c_values:
                token = str(c_value).replace(".", "_")
                yield {"id": f"hog_lr_{penalty}_c_{token}", "family": "logistic_regression", "penalty": penalty, "C": c_value}

    def create(self, candidate, feature_transformer):
        if candidate["family"] == "svm":
            estimator = SVC(
                kernel="linear", C=0.1, class_weight=self.class_weight,
                random_state=self.config.seed,
            )
        elif candidate["family"] == "logistic_regression":
            estimator = LogisticRegression(**self.logistic_parameters(candidate["penalty"], candidate["C"]))
        else:
            raise ValueError("Unsupported candidate family.")
        return Pipeline([
            ("features", clone(feature_transformer)),
            ("scale", StandardScaler()),
            ("classifier", estimator),
        ])
