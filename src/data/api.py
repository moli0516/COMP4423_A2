from .constants import CLASS_NAMES
from .dataset_catalog import DatasetCatalog


def load_dataset(dataset_dir, allow_unverified=False):
    return DatasetCatalog(dataset_dir, allow_unverified).load()


def count_summary(data):
    return {
        split: {
            CLASS_NAMES[int(label)]: {
                "images": sum(row["label"] == label for row in rows),
                "groups": len({row["group_id"] for row in rows if row["label"] == label}),
            }
            for label in ("1", "0")
        }
        for split, rows in data["by_split"].items()
    }
