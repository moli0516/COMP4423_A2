#!/usr/bin/env python3
"""Make single-factor test variants of photos for the batch upload page (diagnostics only).

    python tools/make_variants.py "temp/probe/photo1.jpg" "temp/probe/photo2.jpg" --out temp/variants
    python tools/make_variants.py temp/probe --out temp/variants        # a whole folder

Every variant is written flat as <photo>__<variant>.png, so ONE batch upload covers all photos
and the exported CSV file names still tell you the source and the variant.
These files are test probes: never add them to the dataset (edited copies would leak across splits).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_variants(image: Image.Image) -> dict[str, Image.Image]:
    width, height = image.size
    red, green, blue = image.split()
    return {
        "original": image,
        "gray": image.convert("L").convert("RGB"),
        "swap_rb": Image.merge("RGB", (blue, green, red)),
        "darker": ImageEnhance.Brightness(image).enhance(0.6),
        "brighter": ImageEnhance.Brightness(image).enhance(1.4),
        "blur": image.filter(ImageFilter.GaussianBlur(3)),
        "flip": ImageOps.mirror(image),
        "centre_crop": image.crop((int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8))),
    }


def collect(paths: list[str]) -> list[Path]:
    files = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            print(f"WARNING not found: {path}", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="photo files and/or folders")
    parser.add_argument("--out", type=Path, default=Path("temp/variants"))
    args = parser.parse_args(argv)

    files = collect(args.sources)
    if not files:
        print("ERROR no photos found", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    used, written, failed = set(), 0, 0
    for path in files:
        stem = path.stem.replace("__", "_")
        unique, number = stem, 2
        while unique in used:
            unique, number = f"{stem}_{number}", number + 1
        used.add(unique)
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except OSError as error:
            print(f"WARNING skipped {path}: {error}", file=sys.stderr)
            failed += 1
            continue
        for name, variant in make_variants(image).items():
            variant.save(args.out / f"{unique}__{name}.png")
            written += 1
    print(f"Wrote {written} files for {len(files) - failed} photos to {args.out}")
    print("Next: upload everything in that folder on the Predict page, export the CSV, then run summarise_variants.py")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
