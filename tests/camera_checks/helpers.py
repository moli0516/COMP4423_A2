import numpy as np


def rgb_features(images):
    return np.array([np.asarray(image).mean(axis=(0, 1)) for image in images], dtype=np.float64)
