from inspect import signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.exceptions import TrainingError


def make_logistic_regression(params, seed, balanced):
    """L1 or L2 Logistic Regression; works with both the old (penalty) and new (l1_ratio) scikit-learn API."""
    values = dict(params)
    penalty = values.pop("penalty", "l2")
    if penalty not in ("l1", "l2"):
        raise TrainingError(f"Logistic Regression penalty must be l1 or l2, got {penalty!r}")
    accepted = signature(LogisticRegression).parameters
    penalty_parameter = accepted.get("penalty")
    if penalty_parameter is None or penalty_parameter.default == "deprecated":
        if "l1_ratio" not in accepted:
            raise TrainingError("Unsupported LogisticRegression regularisation API.")
        values["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        values["penalty"] = penalty
    values.setdefault("solver", "liblinear")
    return LogisticRegression(class_weight="balanced" if balanced else None, random_state=seed, **values)


def make_classifier(name, params, seed, balanced, n_jobs):
    if name == "svm":
        return SVC(class_weight="balanced" if balanced else None, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample" if balanced else None,
            random_state=seed, n_jobs=n_jobs, **params,
        )
    if name == "logistic_regression":
        return make_logistic_regression(params, seed, balanced)
    raise TrainingError(f"Unknown classifier: {name}")
