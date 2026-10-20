import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.color import rgb2gray
from skimage.feature import hog


class HogTestTransformer(TransformerMixin, BaseEstimator):
    """Real HOG test backend; production uses the existing ImageFeatureExtractor."""

    def __init__(self, image_size=64, cell=16):
        self.image_size = image_size
        self.cell = cell

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = []
        for source in X:
            if isinstance(source, Image.Image):
                image = source.copy()
            else:
                with Image.open(source) as opened:
                    image = opened.copy()
            image = np.asarray(image.convert("RGB").resize((self.image_size,) * 2))
            vector = hog(rgb2gray(image), orientations=9, pixels_per_cell=(self.cell,) * 2, cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)
