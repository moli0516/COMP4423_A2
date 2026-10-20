import csv
from pathlib import Path

import numpy as np
from PIL import Image


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images, locations = [], []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            group = f"{split}_{label}"
            locations.append({
                "group_id": group, "class_name": "outside" if label == 0 else "inside",
                "boundary_verified": "yes",
            })
            for index in range(3):
                name = f"{group}_{index}.png"
                if split == "test":
                    (root / name).write_bytes(b"not a decodable image")
                else:
                    pixels = np.zeros((80, 96, 3), dtype=np.uint8)
                    pixels[:, :, label] = 170 + index * 10
                    pixels[15:40, 20:60, :] = 40 + index * 15
                    Image.fromarray(pixels).save(root / name)
                images.append({
                    "image_id": f"{group}_{index}", "relative_path": name,
                    "label": str(label), "group_id": group, "split": split,
                })
    write_csv(root / "images.csv", images)
    write_csv(root / "locations.csv", locations)
    return images, locations
