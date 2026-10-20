import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureExperimentConfig:
    dataset_dir: Path
    reference_model: Path
    reference_metadata: Path
    reference_result: Path
    plan_dir: Path
    experiment_id: str = "POLYU-FEATURE-01"
    allow_unverified: bool = True
    camera_diagnostic_status: str = "pending"

    def __post_init__(self):
        for name in ("dataset_dir", "reference_model", "reference_metadata", "reference_result", "plan_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.experiment_id != "POLYU-FEATURE-01":
            raise ValueError("Keep this experiment ID fixed; change plan_dir for a new preparation attempt.")
        if type(self.allow_unverified) is not bool:
            raise ValueError("allow_unverified must be a boolean.")
        if self.camera_diagnostic_status not in {"pending", "completed"}:
            raise ValueError("camera_diagnostic_status must be pending or completed.")
        if self.plan_dir == self.dataset_dir or self.dataset_dir in self.plan_dir.parents:
            raise ValueError("Store plans outside the dataset directory.")
        if self.plan_dir == self.reference_model.parent or self.plan_dir in self.reference_model.parents or self.reference_model.parent in self.plan_dir.parents:
            raise ValueError("Do not place the plan directory above or inside the reference best-model location.")

    @classmethod
    def from_json(cls, path, project_root):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object.")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for name in ("dataset_dir", "reference_model", "reference_metadata", "reference_result", "plan_dir"):
            value = data.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing or invalid path: {name}")
            value = Path(value).expanduser()
            data[name] = value if value.is_absolute() else Path(project_root) / value
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        for name in ("dataset_dir", "reference_model", "reference_metadata", "reference_result", "plan_dir"):
            data[name] = str(data[name])
        return data
