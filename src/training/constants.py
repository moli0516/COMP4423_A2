FEATURE_SETS = {
    "color": {"use_hog": False},
    "hog": {"use_color": False},
    "color_hog": {},
    "color_hog_lbp": {"use_lbp": True},
}
CLASSIFIERS = ("svm", "random_forest")
DEFAULT_GRIDS = {
    "svm": [
        {"kernel": ["rbf"], "C": [1, 10, 100], "gamma": ["scale"]},
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
    ],
    "random_forest": [{
        "n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 2],
    }],
}
SELECTION_RULE = (
    "validation macro-F1, then balanced accuracy, then accuracy; "
    "earlier trial wins exact ties"
)
