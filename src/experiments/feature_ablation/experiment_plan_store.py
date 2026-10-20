import hashlib
import json
from pathlib import Path

from .feature_experiment_planner import sha256_file


class ExperimentPlanStore:
    """Persist a protocol snapshot; never write fitted models or overwrite plans."""

    def __init__(self, plan_dir):
        self.plan_dir = Path(plan_dir)

    def freeze(self, prepared):
        if self.plan_dir.exists():
            raise FileExistsError(f"Plan directory already exists: {self.plan_dir}")
        plan = prepared["plan"]
        if plan.get("training_implemented") is not False or plan.get("completed_fits") != 0:
            raise ValueError("This store accepts preparation-only plans.")
        for item in plan["input_files"].values():
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("An input file changed after preparation; prepare again.")
        dataset = Path(plan["config"]["dataset_dir"])
        for relative, digest in plan["train_validation_image_hashes"].items():
            if sha256_file(dataset / relative) != digest:
                raise ValueError("A train/validation image changed after preparation.")
        for name, contents in prepared["snapshots"].items():
            if hashlib.sha256(contents).hexdigest() != plan["input_files"][name]["sha256"]:
                raise ValueError("CSV snapshot digest differs from the plan.")
        self.plan_dir.mkdir(parents=True, exist_ok=False)
        snapshot_dir = self.plan_dir / "snapshots"
        snapshot_dir.mkdir()
        for name, contents in prepared["snapshots"].items():
            (snapshot_dir / ("images.csv" if name == "images_csv" else "locations.csv")).write_bytes(contents)
        (self.plan_dir / "experiment_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (self.plan_dir / "config.json").write_text(json.dumps(plan["config"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        variants = self.plan_dir / "variants"
        variants.mkdir()
        for variant in plan["variants"]:
            (variants / (variant["variant_id"] + ".json")).write_text(json.dumps(variant, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return self.plan_dir
