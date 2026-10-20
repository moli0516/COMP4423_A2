#!/usr/bin/env python3
"""Check whether how the photos were captured/processed differs between the two classes.

    python tools/audit_sources.py --dataset-dir dataset_dev

For every location group it prints the file-name style (e.g. IMG_ originals vs photo_ messenger exports),
median size in megapixels, bytes per pixel (compression strength) and a sharpness score, then compares
the two classes. If one class was mostly captured/exported one way, a pixel-based classifier can learn that
instead of the scene. Opens train and validation images only (the test split stays untouched).
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def sharpness(image: Image.Image) -> float:
    """Variance of a Laplacian on a 256x256 grey copy: higher = more fine detail."""
    gray = np.asarray(ImageOps.exif_transpose(image).convert("L").resize((256, 256)), dtype=np.float64)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def style(name: str) -> str:
    stem = Path(name).name
    for label, pattern in (("IMG_", r"^IMG_"), ("photo_", r"^photo_"), ("kaboompics", r"^kaboompics")):
        if re.match(pattern, stem, re.I):
            return label
    return "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_dev"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args(argv)

    with (args.dataset_dir / "images.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] in args.splits]
    groups = defaultdict(list)
    for row in rows:
        path = args.dataset_dir / row["relative_path"]
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
            has_exif = bool(image.getexif())
            value = sharpness(image)
        groups[row["group_id"]].append({
            "class": "inside" if str(row["label"]).strip() == "1" else "outside", "split": row["split"],
            "style": style(path.name), "mp": width * height / 1e6, "bpp": path.stat().st_size / (width * height),
            "sharp": value, "exif": has_exif})

    print(f"{'group':8} {'class':8} {'split':10} {'n':>3} {'file style':14} {'MP':>5} {'B/px':>5} {'sharp':>7} {'EXIF':>5}")
    per_class = defaultdict(lambda: defaultdict(list))
    for group in sorted(groups, key=lambda g: (groups[g][0]["class"], g)):
        items = groups[group]
        styles = ",".join(sorted({i["style"] for i in items}))
        med = lambda key: st.median(i[key] for i in items)
        print(f"{group:8} {items[0]['class']:8} {items[0]['split']:10} {len(items):>3} {styles:14} {med('mp'):5.1f} "
              f"{med('bpp'):5.2f} {med('sharp'):7.0f} {sum(i['exif'] for i in items) / len(items):5.0%}")
        for item in items:
            for key in ("mp", "bpp", "sharp"):
                per_class[item["class"]][key].append(item[key])
            per_class[item["class"]]["style"].append(item["style"])

    print("\nPer class (median over images):")
    for name in ("inside", "outside"):
        data = per_class[name]
        if not data["mp"]:
            continue
        counts = {s: data["style"].count(s) for s in sorted(set(data["style"]))}
        print(f"  {name:8} MP {st.median(data['mp']):.1f} | B/px {st.median(data['bpp']):.2f} | "
              f"sharpness {st.median(data['sharp']):.0f} | file styles {counts}")
    print("Large class differences here mean the classifier could separate classes by capture pipeline instead of scene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
