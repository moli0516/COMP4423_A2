#!/usr/bin/env python3
"""Show what the model actually receives under each resize mode (one row per photo).

    python tools/preview_modes.py temp/probe/photo1.jpg temp/probe/photo2.jpg
    -> outputs/diagnostics/preprocessing_modes.png   (original | squash | center_crop | pad)

Uses your real ImagePreprocessor, so the pictures are exactly the arrays fed to the feature extractor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.preprocessing import ImagePreprocessor, PreprocessingConfig
except ImportError:  # adjust here if your package layout differs
    from src.preprocessing.image_preprocessor import ImagePreprocessor
    from src.preprocessing.preprocessing_config import PreprocessingConfig

MODES = ("squash", "center_crop", "pad")
CELL, LABEL_HEIGHT = 256, 18


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/diagnostics/preprocessing_modes.png"))
    args = parser.parse_args(argv)

    processors = {m: ImagePreprocessor(PreprocessingConfig(image_size=CELL, resize_mode=m)) for m in MODES}
    columns = ("original",) + MODES
    sheet = Image.new("RGB", (CELL * len(columns), (CELL + LABEL_HEIGHT) * len(args.photos)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, photo in enumerate(args.photos):
        top = row * (CELL + LABEL_HEIGHT)
        with Image.open(photo) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        thumb = original.copy()
        thumb.thumbnail((CELL, CELL))
        sheet.paste(thumb, ((CELL - thumb.width) // 2, top + LABEL_HEIGHT + (CELL - thumb.height) // 2))
        draw.text((4, top + 2), f"{photo.name[:34]}  {original.width}x{original.height}", fill="black")
        for column, mode in enumerate(MODES, start=1):
            array = processors[mode].load(photo)
            sheet.paste(Image.fromarray(array), (column * CELL, top + LABEL_HEIGHT))
            draw.text((column * CELL + 4, top + 2), mode, fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"Saved {args.out} ({len(args.photos)} photos x {len(columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
