import argparse

from src.preprocessing import ImagePreprocessor, PreprocessingConfig, RESIZE_MODES
from .feature_config import FeatureConfig
from .visual_feature_extractor import VisualFeatureExtractor
from .api import feature_block_sizes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect image feature dimensions.")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--lbp", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=RESIZE_MODES, default="squash")
    parser.add_argument("--pad-value", type=int, default=0)
    parser.add_argument("--hog-cell", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        preprocessing = PreprocessingConfig(args.image_size, args.resize_mode, args.pad_value)
        config = FeatureConfig(use_lbp=args.lbp, hog_pixels_per_cell=args.hog_cell)
        sizes = feature_block_sizes(config, preprocessing)
    except ValueError as error:
        parser.error(str(error))
    print(f"blocks={sizes}, total={sum(sizes.values())}")
    processor, extractor = ImagePreprocessor(preprocessing), VisualFeatureExtractor(config)
    code = 0
    for source in args.images:
        try:
            vector = extractor.extract(processor.load(source))
            print(f"OK {source}: shape={vector.shape}, dtype={vector.dtype}")
        except ValueError as error:
            print(f"ERROR {source}: {error}")
            code = 1
    return code
