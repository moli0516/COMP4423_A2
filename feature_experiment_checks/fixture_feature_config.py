from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixtureFeatureConfig:
    use_color: bool = False
    color_bins: int = 16
    color_grid: int = 1
    use_hog: bool = True
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 32
    hog_cells_per_block: int = 2
    use_lbp: bool = False
    lbp_points: int = 8
    lbp_radius: int = 1

    def to_dict(self):
        return asdict(self)
