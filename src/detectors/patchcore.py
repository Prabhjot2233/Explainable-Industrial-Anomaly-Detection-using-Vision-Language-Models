"""Differentiable PatchCore baseline (WideResNet50-2 layer2+layer3 memory bank)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.data.mvtec import CLIP_MEAN, CLIP_STD
from src.detectors.base import BaseDetector, gaussian_blur, smooth_max

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchCoreDetector(BaseDetector):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        pretrained: bool = True,
        coreset_ratio: float = 0.1,
        max_memory: int = 2000,
        sigma: float = 4.0,
        score_mode: str = "map_max",
        seed: int = 0,
        image_size: int = 240,
    ) -> None:
        super().__init__(score_mode=score_mode, device=device)
        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        net = wide_resnet50_2(weights=weights).float().eval()
        net.requires_grad_(False)
        self.backbone = net.to(self.device)
        self.coreset_ratio = coreset_ratio
        self.max_memory = max_memory
        self.sigma = sigma
        self.seed = seed
        self.image_size = image_size
        self.memory_bank: Tensor | None = None
        for name, val in [("clip_mean", CLIP_MEAN), ("clip_std", CLIP_STD),
                          ("in_mean", IMAGENET_MEAN), ("in_std", IMAGENET_STD)]:
            self.register_buffer(name, torch.tensor(val, device=self.device).view(1, 3, 1, 1))

    def _features(self, x: Tensor) -> Tensor:
        """x CLIP-normalised [B,3,H,W] -> unit-norm patch features [B, 1536, h, w]."""
        x = (x * self.clip_std + self.clip_mean - self.in_mean) / self.in_std
        n = self.backbone
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        f2 = n.layer2(n.layer1(x))
        f3 = n.layer3(f2)
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return F.normalize(torch.cat([f2, f3], dim=1), dim=1)

    def set_reference(self, normal_images: Tensor | None) -> None:
        if normal_images is None:
            self.memory_bank = None
            return
        with torch.no_grad():
            f = self._features(normal_images.to(self.device))
            patches = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
            n = patches.shape[0]
            m = min(n, self.max_memory, max(1, int(round(n * self.coreset_ratio))))
            if m < n:
                g = torch.Generator().manual_seed(self.seed)
                idx = torch.randperm(n, generator=g)[:m].to(self.device)
                patches = patches[idx]
            self.memory_bank = patches.detach()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore needs set_reference(normal_images) before forward().")
        f = self._features(x)
        b, c, h, w = f.shape
        p = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
        cos = p @ self.memory_bank.t()  # [B, hw, M]
        d2 = (2.0 - 2.0 * cos.amax(dim=-1)).clamp(min=1e-8)  # unit vectors: ||a-b||^2
        dist = d2.sqrt().view(b, 1, h, w)
        amap = F.interpolate(dist, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        amap = gaussian_blur(amap, self.sigma)
        if self.score_mode == "combined_smooth":
            score = smooth_max(amap)
        else:
            score = amap.flatten(1).amax(dim=1)
        return score, amap
