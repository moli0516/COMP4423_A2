#!/usr/bin/env python3
"""Show which TRAINING images a photo looks most like, in the model's own feature space.

    python tools/nearest_train.py --pipeline outputs/polyu_lr_01/best/pipeline.joblib \
        --dataset-dir dataset_dev  photo1.jpg photo2.jpg  --k 5

If the nearest training images are mostly campus (inside) scenes of a similar kind, the wrong answer
comes from what the training set contains (coverage), not from a bug. If the photo is far from every
training image, the model is extrapolating and its score should not be trusted.
Diagnostics only: nothing is written, and only train images are opened by default.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets joblib import src.features / src.preprocessing


def features(pipeline, items) -> np.ndarray:
    """Run every step except the classifier (same inputs the classifier sees)."""
    values = list(items)
    for _, step in pipeline.steps[:-1]:
        if step is not None and step != "passthrough":
            values = step.transform(values)
    return np.asarray(values, dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_dev"))
    parser.add_argument("--splits", nargs="+", default=["train"], help="which splits to search (default: train only)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    with (args.dataset_dir / "images.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] in args.splits]
    if len(rows) < 2:
        print(f"ERROR fewer than 2 images in splits {args.splits}", file=sys.stderr)
        return 1
    pipeline = joblib.load(args.pipeline)
    reference = features(pipeline, [args.dataset_dir / r["relative_path"] for r in rows])

    gaps = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    typical = float(np.median(gaps.min(axis=1)))
    print(f"{len(rows)} reference images from {args.splits}; typical nearest-neighbour distance between them: {typical:.2f}")

    for photo in args.photos:
        query = features(pipeline, [photo.read_bytes()])[0]
        distance = np.linalg.norm(reference - query, axis=1)
        order = np.argsort(distance)[: args.k]
        print(f"\n{photo.name}: nearest distance {distance[order[0]]:.2f} = {distance[order[0]] / typical:.1f}x typical")
        for rank, index in enumerate(order, start=1):
            row = rows[index]
            name = "inside" if str(row["label"]).strip() == "1" else "outside"
            print(f"  {rank}. d={distance[index]:.2f}  {name:7} {row['group_id']:8} {row['relative_path']}")
        votes = Counter("inside" if str(rows[i]["label"]).strip() == "1" else "outside" for i in order)
        print(f"  nearest {len(order)}: {votes['inside']} inside, {votes['outside']} outside")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
