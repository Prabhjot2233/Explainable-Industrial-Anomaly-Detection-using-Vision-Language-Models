"""Zero-shot experiment config: dataclass + YAML load/save."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Union

import yaml

VALID_SCORE_MODES = {"global", "map_max", "combined", "combined_smooth"}


@dataclass
class ZeroShotConfig:
    model: str = "ViT-B-16-plus-240"
    pretrained: str = "laion400m_e32"
    data_root: str = "data/mvtec_ad"
    results_dir: str = "results"
    categories: List[str] = field(
        default_factory=lambda: ["carpet", "leather", "tile", "bottle", "transistor"]
    )
    image_size: int = 240
    window_scales: List[int] = field(default_factory=lambda: [2, 3])
    window_chunk_size: int = 16
    batch_size: int = 1
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    score_mode: str = "combined"
    smooth_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.score_mode not in VALID_SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {sorted(VALID_SCORE_MODES)}, got '{self.score_mode}'"
            )

    @property
    def data_path(self) -> Path:
        return Path(self.data_root)

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir)


PathLike = Union[str, Path]


def save_config(cfg: ZeroShotConfig, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def load_config(path: PathLike) -> ZeroShotConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in fields(ZeroShotConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return ZeroShotConfig(**data)
