from sklearn.base import BaseEstimator, TransformerMixin


class NoDecodeTransformer(TransformerMixin, BaseEstimator):
    """Fail if architecture preparation ever tries to fit or decode features."""

    def __init__(self, preprocessing_config=None, feature_config=None):
        self.preprocessing_config = preprocessing_config
        self.feature_config = feature_config

    def fit(self, X, y=None):
        raise AssertionError("Preparation must not fit feature transformers.")

    def transform(self, X):
        raise AssertionError("Preparation must not decode images or extract vectors.")
