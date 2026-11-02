"""MVTec AD dataset loader: 240x240 CLIP-normalised images + binary masks."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def clip_normalize(img01: Tensor) -> Tensor:
    """[..., 3, H, W] in [0,1] -> CLIP-normalised."""
    mean = torch.tensor(CLIP_MEAN, device=img01.device, dtype=img01.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, device=img01.device, dtype=img01.dtype).view(3, 1, 1)
    return (img01 - mean) / std


def clip_denormalize(img: Tensor) -> Tensor:
    """Inverse of clip_normalize (pure torch, differentiable)."""
    mean = torch.tensor(CLIP_MEAN, device=img.device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, device=img.device, dtype=img.dtype).view(3, 1, 1)
    return img * std + mean


class MVTecDataset(Dataset):
    """Expected layout: <root>/<category>/{train/good, test/<defect>, ground_truth/<defect>}.

    Each item is a dict: image [3,S,S] float32, mask [1,S,S] float32 {0,1},
    label int (0 normal / 1 anomalous), defect_type str, path str.
    """

    def __init__(
        self,
        root: Union[str, Path],
        category: str,
        split: str = "test",
        image_size: int = 240,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        cat_dir = self.root / category / split
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"MVTec folder not found: {cat_dir.as_posix()}")

        self.samples: list[tuple[Path, Path | None, int, str]] = []
        for defect_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                if defect == "good":
                    self.samples.append((img_path, None, 0, defect))
                else:
                    mask_path = (
                        self.root / category / "ground_truth" / defect / f"{img_path.stem}_mask.png"
                    )
                    self.samples.append((img_path, mask_path if mask_path.exists() else None, 1, defect))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, label, defect = self.samples[idx]
        s = self.image_size
        img = Image.open(img_path).convert("RGB").resize((s, s), Image.BICUBIC)
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        img_t = clip_normalize(img_t)
        if mask_path is not None:
            m = Image.open(mask_path).convert("L").resize((s, s), Image.NEAREST)
            mask = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]
        else:
            mask = torch.zeros(1, s, s, dtype=torch.float32)
        return {
            "image": img_t,
            "mask": mask,
            "label": label,
            "defect_type": defect,
            "path": img_path.as_posix(),
        }
