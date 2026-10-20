import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

from .feature_config import FeatureConfig


class VisualFeatureExtractor:
    """HSV, luminance HOG and uniform LBP features from uint8 RGB pixels."""

    def __init__(self, config=None):
        self.config = config if config is not None else FeatureConfig()

    @staticmethod
    def _validate(image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Expected a NumPy uint8 RGB image.")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a uint8 RGB image with shape (height, width, 3).")
        if min(image.shape[:2]) < 1:
            raise ValueError("Image dimensions must be nonempty.")
        return image

    def block_sizes(self, image_shape):
        height, width = image_shape[:2]
        config = self.config
        sizes = {}
        if min(height, width) < 1:
            raise ValueError("Image dimensions must be nonempty.")
        if config.use_color:
            sizes["color"] = 3 * config.color_bins * config.color_grid ** 2
        if config.use_hog:
            cells_y = height // config.hog_pixels_per_cell
            cells_x = width // config.hog_pixels_per_cell
            block = config.hog_cells_per_block
            if min(cells_y, cells_x) < block:
                raise ValueError("Image is too small for the configured HOG blocks.")
            sizes["hog"] = (cells_y - block + 1) * (cells_x - block + 1) * block ** 2 * config.hog_orientations
        if config.use_lbp:
            sizes["lbp"] = config.lbp_points + 2
        return sizes

    def color_histogram_features(self, image):
        image = self._validate(image)
        hsv = rgb2hsv(image)
        bins, grid = self.config.color_bins, self.config.color_grid
        vectors = []
        for row in np.array_split(hsv, grid, axis=0):
            for tile in np.array_split(row, grid, axis=1):
                for channel in range(3):
                    histogram, _ = np.histogram(tile[:, :, channel], bins=bins, range=(0.0, 1.0))
                    histogram = histogram.astype(np.float32)
                    total = histogram.sum()
                    if total:
                        histogram /= total
                    vectors.append(histogram)
        return np.concatenate(vectors).astype(np.float32)

    def hog_features(self, image):
        image = self._validate(image)
        config = self.config
        cell, block = config.hog_pixels_per_cell, config.hog_cells_per_block
        if min(image.shape[:2]) // cell < block:
            raise ValueError("Image is too small for the configured HOG blocks.")
        values = hog(
            rgb2gray(image), orientations=config.hog_orientations,
            pixels_per_cell=(cell, cell), cells_per_block=(block, block),
            block_norm="L2-Hys", feature_vector=True,
        )
        return values.astype(np.float32)

    def lbp_features(self, image):
        image = self._validate(image)
        gray = np.rint(rgb2gray(image) * 255).astype(np.uint8)
        points, radius = self.config.lbp_points, self.config.lbp_radius
        values = local_binary_pattern(gray, points, radius, method="uniform")
        histogram, _ = np.histogram(values, bins=points + 2, range=(0, points + 2))
        return histogram.astype(np.float32) / gray.size

    def extract_blocks(self, image):
        image = self._validate(image)
        blocks = {}
        if self.config.use_color:
            blocks["color"] = self.color_histogram_features(image)
        if self.config.use_hog:
            blocks["hog"] = self.hog_features(image)
        if self.config.use_lbp:
            blocks["lbp"] = self.lbp_features(image)
        return blocks

    def extract(self, image):
        vector = np.concatenate(list(self.extract_blocks(image).values())).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite feature values.")
        return vector
