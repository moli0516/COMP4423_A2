from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FeatureVariant:
    variant_id: str
    description: str
    use_color: bool
    use_lbp: bool
    color_grid: int

    def apply(self, base_config):
        return replace(
            base_config, use_color=self.use_color, use_hog=True,
            use_lbp=self.use_lbp, color_grid=self.color_grid,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def planned_variants(cls):
        return (
            cls("F0", "HOG baseline", False, False, 1),
            cls("F1", "HOG + global HSV", True, False, 1),
            cls("F2", "HOG + 2x2 spatial HSV", True, False, 2),
            cls("F3", "HOG + uniform LBP", False, True, 1),
        )
