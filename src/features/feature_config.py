from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class FeatureConfig:
    use_color: bool = True
    color_bins: int = 16
    color_grid: int = 1
    use_hog: bool = True
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 32
    hog_cells_per_block: int = 2
    use_lbp: bool = False
    lbp_points: int = 8
    lbp_radius: int = 1

    def __post_init__(self):
        for name in ("use_color", "use_hog", "use_lbp"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        if not (self.use_color or self.use_hog or self.use_lbp):
            raise ValueError("Enable at least one feature type.")
        for name in (
            "color_bins", "color_grid", "hog_orientations", "hog_pixels_per_cell",
            "hog_cells_per_block", "lbp_points", "lbp_radius",
        ):
            value = getattr(self, name)
            minimum = 2 if name == "color_bins" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, Mapping):
            raise ValueError("Feature settings must be a mapping.")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown feature settings: {sorted(map(str, unknown))}")
        return cls(**dict(values))


DEFAULT_FEATURE_CONFIG = FeatureConfig()
