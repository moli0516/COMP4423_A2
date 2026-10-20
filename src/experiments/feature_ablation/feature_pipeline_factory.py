from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class FeaturePipelineFactory:
    """Clone the existing LR model settings, changing only the feature configuration."""

    def __init__(self, reference_pipeline, dimension_function=None):
        self.reference = reference_pipeline
        if not isinstance(reference_pipeline, Pipeline):
            raise ValueError("Reference must be a complete sklearn Pipeline.")
        if list(reference_pipeline.named_steps) != ["features", "scale", "classifier"]:
            raise ValueError("Expected features -> scale -> classifier.")
        features = reference_pipeline.named_steps["features"]
        if not hasattr(features, "feature_config") or not hasattr(features, "preprocessing_config"):
            raise ValueError("Reference feature transformer lacks its configuration.")
        if not isinstance(reference_pipeline.named_steps["scale"], StandardScaler):
            raise ValueError("Reference scaler must be StandardScaler.")
        classifier = reference_pipeline.named_steps["classifier"]
        if not isinstance(classifier, LogisticRegression):
            raise ValueError("Reference must be the selected logistic-regression model, not the old SVM.")
        check_is_fitted(classifier)
        if list(classifier.classes_) != [0, 1]:
            raise ValueError("Reference classes must be outside=0 and inside=1.")
        parameters = classifier.get_params(deep=False)
        if parameters.get("C") != 0.01 or parameters.get("solver") != "liblinear":
            raise ValueError("Reference must be the retained L2 LR C=0.01, solver=liblinear.")
        if parameters.get("max_iter") != 10000 or parameters.get("tol") != 1e-4 or parameters.get("random_state") != 42:
            raise ValueError("Reference numerical settings do not match POLYU-LR-01.")
        penalty = parameters.get("penalty", "deprecated")
        if penalty == "l2":
            pass
        elif penalty == "deprecated" and parameters.get("l1_ratio") == 0:
            pass
        else:
            raise ValueError("Reference estimator must use effective L2 regularization.")
        base_config = features.feature_config
        if base_config.use_color or not base_config.use_hog or base_config.use_lbp:
            raise ValueError("Reference feature configuration must be HOG-only.")
        if dimension_function is None:
            from src.features.api import feature_dimension
            dimension_function = feature_dimension
        self.dimension_function = dimension_function

    def create(self, variant):
        features = clone(self.reference.named_steps["features"])
        features.set_params(feature_config=variant.apply(features.feature_config))
        return Pipeline([
            ("features", features),
            ("scale", clone(self.reference.named_steps["scale"])),
            ("classifier", clone(self.reference.named_steps["classifier"])),
        ])

    def describe(self, variant):
        pipeline = self.create(variant)
        features = pipeline.named_steps["features"]
        return {
            **variant.to_dict(),
            "preprocessing_config": features.preprocessing_config.to_dict(),
            "feature_config": features.feature_config.to_dict(),
            "feature_dimension": int(self.dimension_function(features.feature_config, features.preprocessing_config)),
            "estimator_parameters": pipeline.named_steps["classifier"].get_params(deep=False),
            "pipeline_steps": list(pipeline.named_steps),
            "fitted": False,
        }
