#!/usr/bin/env python3
"""Summarise a batch-upload CSV of variants made by make_variants.py.

    python tools/summarise_variants.py predictions_lr.csv predictions_feature.csv

Prints, per model, one row per photo and one column per variant (I = inside, O = outside, then the score).
A star marks a variant whose label differs from the original photo. The flip rate per variant shows
which factor the model reacts to. Also writes outputs/diagnostics/variant_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ["original", "gray", "swap_rb", "darker", "brighter", "blur", "flip", "centre_crop"]
PATTERN = re.compile(r"^(?P<source>.+)__(?P<variant>[a-z_]+)\.png$")
MEANING = {
    "gray": "colour reliance", "swap_rb": "colour/hue reliance", "darker": "lighting sensitivity",
    "brighter": "lighting sensitivity", "blur": "fine-texture reliance (e.g. brick joints)",
    "flip": "position-specific features", "centre_crop": "background/periphery reliance",
}


def parse_score(text: str):
    text = (text or "").strip().lstrip("'")  # older app versions prefixed negative numbers with an apostrophe
    try:
        return float(text)
    except ValueError:
        return None


def load(paths: list[Path]) -> dict:
    data = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                match = PATTERN.match(Path(row.get("file_name", "")).name)
                if not match or row.get("status") != "done" or match["variant"] not in VARIANTS:
                    continue
                data[row.get("model", "?")][match["source"]][match["variant"]] = (int(row["label"]), parse_score(row.get("score")))
    return data


def cell(entry, original) -> str:
    if entry is None:
        return "-"
    label, score = entry
    text = f"{'I' if label == 1 else 'O'} {score:+.2f}" if score is not None else ("I" if label == 1 else "O")
    return text + ("*" if original is not None and label != original[0] else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/diagnostics/variant_summary.csv"))
    args = parser.parse_args(argv)
    data = load(args.csv_files)
    if not data:
        print("ERROR no rows named <photo>__<variant>.png with status done were found", file=sys.stderr)
        return 1

    records = []
    for model in sorted(data):
        sources = data[model]
        print(f"\nModel: {model}")
        width = max(len(s) for s in sources) + 2
        print("photo".ljust(width) + "".join(v.ljust(14) for v in VARIANTS))
        flips, totals = defaultdict(int), defaultdict(int)
        for source in sorted(sources):
            original = sources[source].get("original")
            print(source.ljust(width) + "".join(cell(sources[source].get(v), original).ljust(14) for v in VARIANTS))
            for variant in VARIANTS:
                entry = sources[source].get(variant)
                flipped = bool(original and entry and entry[0] != original[0])
                if variant != "original" and original and entry:
                    totals[variant] += 1
                    flips[variant] += flipped
                if entry:
                    records.append([model, source, variant, entry[0], entry[1], int(flipped)])
        print("\nFlip rate vs original (a flip means the label changed although the scene did not):")
        for variant in VARIANTS[1:]:
            if totals[variant]:
                print(f"  {variant:12} {flips[variant]}/{totals[variant]}  -> {MEANING[variant]}")
        if "original" not in {v for s in sources.values() for v in s}:
            print("  (no original files found, so flips cannot be computed)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "source", "variant", "label", "score", "flipped_vs_original"])
        writer.writerows(records)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
