from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.exceptions import TrainingError


def make_classifier(name, params, seed, balanced, n_jobs):
    if name == "svm":
        return SVC(class_weight="balanced" if balanced else None, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample" if balanced else None,
            random_state=seed, n_jobs=n_jobs, **params,
        )
    raise TrainingError(f"Unknown classifier: {name}")
