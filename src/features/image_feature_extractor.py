import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing import ImagePreprocessor, PreprocessingConfig
from .feature_config import FeatureConfig
from .visual_feature_extractor import VisualFeatureExtractor


class ImageFeatureExtractor(TransformerMixin, BaseEstimator):
    """Stateless sklearn transformer for paths, encoded bytes, streams and PIL sources."""

    def __init__(self, preprocessing_config=None, feature_config=None):
        self.preprocessing_config = preprocessing_config
        self.feature_config = feature_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, (str, os.PathLike, bytes, bytearray)):
            raise ValueError("Pass a sequence of image sources, not a single source.")
        preprocessing = self.preprocessing_config or PreprocessingConfig()
        feature_config = self.feature_config or FeatureConfig()
        processor = ImagePreprocessor(preprocessing)
        extractor = VisualFeatureExtractor(feature_config)
        vectors = [extractor.extract(processor.load(source)) for source in X]
        if not vectors:
            size = preprocessing.image_size
            dimension = sum(extractor.block_sizes((size, size, 3)).values())
            return np.empty((0, dimension), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)
