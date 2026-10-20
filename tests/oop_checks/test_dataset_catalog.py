import tempfile
import unittest
from pathlib import Path

from tests.oop_checks.helpers import make_dataset, write_csv
from src.data import DatasetCatalog
from src.exceptions import TrainingError


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images, self.locations = make_dataset(self.root)

    def rewrite_images(self):
        write_csv(self.root / "images.csv", self.images)

    def test_valid_catalog_does_not_decode_test(self):
        result = DatasetCatalog(self.root).load()
        self.assertEqual(len(result["by_split"]["test"]), 6)
        self.assertEqual(len(result["images_csv_sha256"]), 64)

    def test_duplicate_image_id_rejected(self):
        self.images[1]["image_id"] = self.images[0]["image_id"]
        self.rewrite_images()
        with self.assertRaisesRegex(TrainingError, "duplicate image_id"):
            DatasetCatalog(self.root).load()

    def test_duplicate_image_path_rejected(self):
        self.images[1]["relative_path"] = self.images[0]["relative_path"]
        self.rewrite_images()
        with self.assertRaisesRegex(TrainingError, "duplicate relative_path"):
            DatasetCatalog(self.root).load()

    def test_group_split_leakage_rejected(self):
        self.images[1]["split"] = "validation"
        self.rewrite_images()
        with self.assertRaisesRegex(TrainingError, "more than one split"):
            DatasetCatalog(self.root).load()

    def test_group_mixed_labels_rejected(self):
        self.images[1]["label"] = "1"
        self.rewrite_images()
        with self.assertRaisesRegex(TrainingError, "both labels"):
            DatasetCatalog(self.root).load()

    def test_path_escape_rejected(self):
        self.images[0]["relative_path"] = "../escape.png"
        self.rewrite_images()
        with self.assertRaisesRegex(TrainingError, "leaves the dataset directory"):
            DatasetCatalog(self.root).load()

    def test_unverified_guard_and_override(self):
        self.locations[0]["boundary_verified"] = "no"
        write_csv(self.root / "locations.csv", self.locations)
        with self.assertRaisesRegex(TrainingError, "boundary-verified"):
            DatasetCatalog(self.root).load()
        self.assertEqual(len(DatasetCatalog(self.root, True).load()["rows"]), 18)

    def test_missing_image_rejected(self):
        (self.root / self.images[0]["relative_path"]).unlink()
        with self.assertRaisesRegex(TrainingError, "does not exist"):
            DatasetCatalog(self.root).load()
