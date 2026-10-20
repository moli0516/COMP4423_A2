import numpy as np

from src.exceptions import TrainingError
from src.features import VisualFeatureExtractor, feature_block_sizes
from src.preprocessing import ImagePreprocessor
from src.utils.file_io import resolve_image_path


def extract_matrix(rows, dataset_dir, preprocessing_config, feature_config):
    preprocessor = ImagePreprocessor(preprocessing_config)
    extractor = VisualFeatureExtractor(feature_config)
    vectors = []
    for row in rows:
        path = resolve_image_path(dataset_dir, row["relative_path"])
        try:
            vectors.append(extractor.extract(preprocessor.load(path)))
        except ValueError as error:
            raise TrainingError(f"{row['relative_path']}: {error}") from error
    if not vectors:
        raise TrainingError("Cannot extract an empty split.")
    return (
        np.vstack(vectors).astype(np.float32),
        np.array([int(row["label"]) for row in rows], dtype=int),
    )


def feature_columns(all_config, preprocessing_config, selected_config):
    offsets, start = {}, 0
    for name, size in feature_block_sizes(all_config, preprocessing_config).items():
        offsets[name] = (start, start + size)
        start += size
    selected = feature_block_sizes(selected_config, preprocessing_config)
    return np.concatenate([np.arange(*offsets[name]) for name in selected])
