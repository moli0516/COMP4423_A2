#!/usr/bin/env python3
"""List near-duplicate photos inside each location group (it never deletes anything).

    python tools/find_duplicates.py --dataset-dir dataset_dev
    python tools/find_duplicates.py --dataset-dir dataset_dev --threshold 8 --splits train validation

Uses a 64-bit difference hash: photos whose hashes differ in only a few bits look almost the same.
By default only train and validation images are opened (add --include-test to check the test split too).
Review the pairs by eye before removing anything, and re-train afterwards because images.csv changes.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def dhash(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_dev"))
    parser.add_argument("--threshold", type=int, default=6, help="max differing bits (of 64) to call a pair near-duplicate")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args(argv)
    splits = set(args.splits) | ({"test"} if args.include_test else set())

    with (args.dataset_dir / "images.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] in splits]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    print(f"{'group':8} {'split':10} {'images':>6} {'near-dup pairs':>15}")
    details = []
    for group in sorted(groups):
        members = groups[group]
        hashes = [dhash(args.dataset_dir / m["relative_path"]) for m in members]
        pairs = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                distance = int(np.count_nonzero(hashes[i] != hashes[j]))
                if distance <= args.threshold:
                    pairs.append((distance, members[i]["relative_path"], members[j]["relative_path"]))
        print(f"{group:8} {members[0]['split']:10} {len(members):>6} {len(pairs):>15}")
        details += [(group, *pair) for pair in sorted(pairs)]
    if details:
        print(f"\nNear-duplicate pairs (distance <= {args.threshold} of 64 bits):")
        for group, distance, first, second in details:
            print(f"  {group}  d={distance:2}  {first}  <->  {second}")
    else:
        print("\nNo near-duplicate pairs found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
