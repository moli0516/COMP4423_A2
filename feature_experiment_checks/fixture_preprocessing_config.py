from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixturePreprocessingConfig:
    image_size: int = 256
    resize_mode: str = "squash"
    pad_value: int = 0

    def to_dict(self):
        return asdict(self)
