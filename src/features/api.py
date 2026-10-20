from src.preprocessing import PreprocessingConfig
from .visual_feature_extractor import VisualFeatureExtractor


def color_histogram_features(image, config=None):
    return VisualFeatureExtractor(config).color_histogram_features(image)


def hog_features(image, config=None):
    return VisualFeatureExtractor(config).hog_features(image)


def lbp_features(image, config=None):
    return VisualFeatureExtractor(config).lbp_features(image)


def extract_feature_blocks(image, config=None):
    return VisualFeatureExtractor(config).extract_blocks(image)


def extract_features(image, config=None):
    return VisualFeatureExtractor(config).extract(image)


def feature_block_sizes(feature_config=None, preprocessing_config=None):
    preprocessing = preprocessing_config or PreprocessingConfig()
    size = preprocessing.image_size
    return VisualFeatureExtractor(feature_config).block_sizes((size, size, 3))


def feature_dimension(feature_config=None, preprocessing_config=None):
    return sum(feature_block_sizes(feature_config, preprocessing_config).values())
