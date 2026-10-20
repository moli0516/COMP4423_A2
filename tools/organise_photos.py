#!/usr/bin/env python3
"""Organise new photos into the dataset layout in two safe steps.

Step 1 - scan: list the photos, suggest location groups, write an editable manifest.
    python tools/organise_photos.py scan --source temp/images --manifest temp/manifest.csv

Step 2 - fill in the manifest (class, description, boundary_verified, split), then apply:
    python tools/organise_photos.py apply --manifest temp/manifest.csv --dataset-dir dataset_dev --dry-run
    python tools/organise_photos.py apply --manifest temp/manifest.csv --dataset-dir dataset_dev

Notes
- EXIF time and GPS are used only to *suggest* groups and to help you verify the boundary.
  The classifier never sees them (pixels only).
- Originals are copied, never moved or modified.
- New groups may only go to train or validation. The test split is never touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXIF_IFD, GPS_IFD = 0x8769, 0x8825
MANIFEST_FIELDS = ["file", "temp_group", "class", "location_description", "boundary_verified", "split", "exclude",
                   "exif_time", "gps_lat", "gps_lon", "width", "height", "sha256", "duplicate_of", "note"]
GROUP_FIELDS = ("class", "location_description", "boundary_verified", "split")
IMAGE_FIELDS = ["image_id", "relative_path", "label", "group_id", "split"]
LOCATION_FIELDS = ["group_id", "class_name", "location_description", "boundary_verified"]
CLASSES = {"inside": ("in", 1), "outside": ("out", 0)}
ALLOWED_SPLITS = ("train", "validation")
TIME_FORMAT = "%Y:%m:%d %H:%M:%S"


class OrganiseError(Exception):
    pass


def to_degrees(dms, ref) -> float:
    degrees, minutes, seconds = (float(value) for value in dms)
    value = degrees + minutes / 60 + seconds / 3600
    return -value if str(ref).upper() in ("S", "W") else value


def haversine_m(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def read_info(path: Path) -> dict:
    """Return displayed size plus optional EXIF time and GPS (used only for grouping hints)."""
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
    if exif.get(0x0112) in (5, 6, 7, 8):  # rotated by EXIF: report the size people actually see
        width, height = height, width
    taken = None
    raw = exif.get_ifd(EXIF_IFD).get(0x9003) or exif.get(0x0132)
    if raw:
        try:
            taken = datetime.strptime(str(raw).strip(), TIME_FORMAT)
        except ValueError:
            pass
    gps = None
    try:
        data = exif.get_ifd(GPS_IFD)
        if 2 in data and 4 in data:
            gps = (to_degrees(data[2], data.get(1, "N")), to_degrees(data[4], data.get(3, "E")))
    except (TypeError, ValueError, ZeroDivisionError):
        gps = None
    return {"width": width, "height": height, "time": taken, "gps": gps}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(args) -> int:
    source, manifest = Path(args.source), Path(args.manifest)
    if not source.is_dir():
        raise OrganiseError(f"Source folder not found: {source}")
    if manifest.exists() and not args.force:
        raise OrganiseError(f"{manifest} already exists (your edits would be lost). Use --force to overwrite.")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise OrganiseError(f"No images found under {source}")

    records, first_seen = [], {}
    for path in files:
        record = {name: "" for name in MANIFEST_FIELDS}
        record["file"] = path.relative_to(source).as_posix()
        record["_time"], record["_gps"] = None, None
        try:
            info = read_info(path)
        except (OSError, UnidentifiedImageError) as error:
            record["exclude"], record["note"] = "x", f"unreadable: {error}"
            records.append(record)
            continue
        record.update(width=info["width"], height=info["height"], sha256=sha256_of(path))
        record["_time"], record["_gps"] = info["time"], info["gps"]
        if info["time"]:
            record["exif_time"] = info["time"].strftime("%Y-%m-%d %H:%M:%S")
        if info["gps"]:
            record["gps_lat"], record["gps_lon"] = f"{info['gps'][0]:.6f}", f"{info['gps'][1]:.6f}"
        if record["sha256"] in first_seen:
            record["duplicate_of"] = first_seen[record["sha256"]]
            record["note"] = "exact duplicate: skipped by apply"
        else:
            first_seen[record["sha256"]] = record["file"]
        records.append(record)

    dated = sorted((r for r in records if r["_time"]), key=lambda r: (r["_time"], r["file"]))
    undated = [r for r in records if not r["_time"] and r["exclude"] != "x"]
    number, previous = 0, None
    for record in dated:
        new_group = previous is None
        if previous is not None:
            gap = (record["_time"] - previous["_time"]).total_seconds()
            far = bool(record["_gps"] and previous["_gps"] and haversine_m(record["_gps"], previous["_gps"]) > args.gps_metres)
            new_group = gap > args.gap_minutes * 60 or far
        number += new_group
        record["temp_group"] = f"g{number:02d}"
        previous = record
    for record in undated:
        record["temp_group"] = "g_unknown"
        record["note"] = (record["note"] + "; " if record["note"] else "") + "no EXIF time: set temp_group yourself"

    records.sort(key=lambda r: (r["temp_group"] or "~", r["_time"] or datetime.min, r["file"]))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    groups = sorted({r["temp_group"] for r in records if r["temp_group"]})
    print(f"Scanned {len(files)} photos -> {len(groups)} suggested groups. Manifest: {manifest}")
    for group in groups:
        members = [r for r in records if r["temp_group"] == group]
        print(f"  {group}: {len(members)} photos, {members[0]['exif_time'] or 'no time'}")
    skipped = sum(1 for r in records if r["duplicate_of"] or r["exclude"] == "x")
    if skipped:
        print(f"  {skipped} photo(s) flagged as duplicate/unreadable and will be skipped.")
    print("Next: open the manifest, check the groups, and fill class / location_description / boundary_verified / split")
    print("(once per group is enough; put x in 'exclude' for photos with identifiable people or restricted areas).")
    return 0


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def next_numbers(dataset: Path) -> dict:
    used = {"in": 0, "out": 0}
    for name in ("locations.csv", "images.csv"):
        path = dataset / name
        if path.is_file():
            for row in read_csv_rows(path)[1]:
                match = re.fullmatch(r"(in|out)_(\d+)", (row.get("group_id") or "").strip())
                if match:
                    used[match.group(1)] = max(used[match.group(1)], int(match.group(2)))
    return used


def build_plan(args) -> list[dict]:
    manifest, dataset, source = Path(args.manifest), Path(args.dataset_dir), Path(args.source)
    if not manifest.is_file():
        raise OrganiseError(f"Manifest not found: {manifest}")
    rows = read_csv_rows(manifest)[1]
    active = [r for r in rows if not r["exclude"].strip() and not r["duplicate_of"].strip()]
    errors, groups = [], {}
    for row in active:
        group = row["temp_group"].strip()
        if not group:
            errors.append(f"{row['file']}: temp_group is empty (set one or put x in 'exclude')")
            continue
        groups.setdefault(group, []).append(row)

    settings = {}
    for group, members in groups.items():
        values = {}
        for field in GROUP_FIELDS:
            found = {m[field].strip() for m in members if m[field].strip()}
            if len(found) > 1:
                errors.append(f"{group}: conflicting values for '{field}': {sorted(found)}")
            elif not found:
                errors.append(f"{group}: '{field}' is empty")
            values[field] = next(iter(found), "")
        values["class"] = values["class"].lower()
        values["split"] = values["split"].lower()
        if values["class"] and values["class"] not in CLASSES:
            errors.append(f"{group}: class must be inside or outside, got '{values['class']}'")
        if values["split"] == "test":
            errors.append(f"{group}: new groups cannot go to the test split (use train or validation)")
        elif values["split"] and values["split"] not in ALLOWED_SPLITS:
            errors.append(f"{group}: split must be train or validation, got '{values['split']}'")
        settings[group] = values
    if not groups and not errors:
        errors.append("Nothing to organise: every photo is excluded or a duplicate.")
    if errors:
        raise OrganiseError("Manifest problems:\n  - " + "\n  - ".join(errors))

    numbers, plan = next_numbers(dataset), []
    for group, members in groups.items():
        values = settings[group]
        prefix, label = CLASSES[values["class"]]
        numbers[prefix] += 1
        group_id = f"{prefix}_{numbers[prefix]:03d}"
        members = sorted(members, key=lambda r: (r["exif_time"], r["file"]))
        photos = []
        for index, row in enumerate(members, start=1):
            src = source / row["file"]
            if not src.is_file():
                raise OrganiseError(f"Photo missing: {src} (pass --source if the folder moved)")
            relative = f"images/{values['class']}/{group_id}/{group_id}_{index:02d}{src.suffix.lower()}"
            photos.append({"src": src, "relative": relative, "image_id": f"{group_id}_{index:02d}"})
        plan.append({"temp_group": group, "group_id": group_id, "label": label, "photos": photos, **values})
    return plan


def copy_photo(src: Path, dst: Path, strip_gps: bool) -> None:
    if dst.exists():
        raise OrganiseError(f"Refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if strip_gps and src.suffix.lower() in (".jpg", ".jpeg"):
        with Image.open(src) as image:
            exif = image.getexif()
            if GPS_IFD in exif:
                del exif[GPS_IFD]
            options = {"format": "JPEG", "quality": "keep", "exif": exif}
            if image.info.get("icc_profile"):
                options["icc_profile"] = image.info["icc_profile"]
            image.save(dst, **options)
    else:
        shutil.copy2(src, dst)


def append_rows(path: Path, default_header: list[str], required: list[str], rows: list[dict]) -> None:
    header = default_header
    if path.is_file():
        header = read_csv_rows(path)[0]
        missing = [name for name in required if name not in header]
        if missing:
            raise OrganiseError(f"{path} lacks columns {missing}; add them or edit this script.")
        with path.open("rb+") as handle:  # make sure the last existing row ends with a newline
            handle.seek(0, 2)
            if handle.tell() and (handle.seek(-1, 2) or handle.read(1)) not in (b"\n",):
                handle.write(b"\r\n")
    with path.open("a" if path.is_file() else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def apply(args) -> int:
    plan = build_plan(args)
    dataset = Path(args.dataset_dir)
    total = sum(len(item["photos"]) for item in plan)
    print(f"{'DRY RUN - ' if args.dry_run else ''}{total} photos in {len(plan)} new groups -> {dataset}")
    for item in plan:
        print(f"  {item['temp_group']} -> {item['group_id']} [{item['class']}, {item['split']}, "
              f"{len(item['photos'])} photos] {item['location_description']}")
        if item["boundary_verified"].lower() != "yes":
            print(f"     WARNING boundary_verified='{item['boundary_verified']}': training rejects unverified groups by default")
    if args.dry_run:
        print("Nothing was written.")
        return 0

    for item in plan:
        for photo in item["photos"]:
            copy_photo(photo["src"], dataset / photo["relative"], args.strip_gps)
    append_rows(dataset / "locations.csv", LOCATION_FIELDS, LOCATION_FIELDS, [
        {"group_id": i["group_id"], "class_name": i["class"], "location_description": i["location_description"],
         "boundary_verified": i["boundary_verified"]} for i in plan])
    append_rows(dataset / "images.csv", IMAGE_FIELDS, IMAGE_FIELDS, [
        {"image_id": p["image_id"], "relative_path": p["relative"], "label": i["label"], "group_id": i["group_id"],
         "split": i["split"]} for i in plan for p in i["photos"]])
    print("Done. Originals were left untouched.")
    print("Next: do NOT re-run a splitter that re-shuffles existing groups; the splits above are already recorded.")
    print("Then retrain and check the new groups do not overlap scenes already in another split.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    scan_p = sub.add_parser("scan", help="write an editable manifest with suggested groups")
    scan_p.add_argument("--source", default="temp/images")
    scan_p.add_argument("--manifest", default="temp/manifest.csv")
    scan_p.add_argument("--gap-minutes", type=float, default=10, help="new group if photos are further apart in time")
    scan_p.add_argument("--gps-metres", type=float, default=75, help="new group if GPS positions are further apart")
    scan_p.add_argument("--force", action="store_true")
    apply_p = sub.add_parser("apply", help="copy photos into the dataset and append CSV rows")
    apply_p.add_argument("--source", default="temp/images")
    apply_p.add_argument("--manifest", default="temp/manifest.csv")
    apply_p.add_argument("--dataset-dir", default="dataset_dev")
    apply_p.add_argument("--dry-run", action="store_true")
    apply_p.add_argument("--strip-gps", action="store_true", help="remove GPS from copied JPEGs (recommended before submission)")
    args = parser.parse_args(argv)
    try:
        return scan(args) if args.command == "scan" else apply(args)
    except OrganiseError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())