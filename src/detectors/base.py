"""BaseDetector contract shared by every detector."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.utils.config import VALID_SCORE_MODES


class BaseDetector(nn.Module, ABC):
    """forward(x [B,3,240,240]) -> (score [B], anomaly_map [B,240,240]).

    forward() must stay pure PyTorch and never use torch.no_grad(), so that
    gradients can flow back to the input pixels (needed by Week 4 explainers).
    """

    def __init__(self, score_mode: str = "combined", device: torch.device | str = "cpu") -> None:
        super().__init__()
        if score_mode not in VALID_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {sorted(VALID_SCORE_MODES)}")
        self.score_mode = score_mode
        self.device = torch.device(device)

    def set_reference(self, normal_images: Tensor | None) -> None:
        """Optional few-shot hook. Zero-shot detectors ignore it."""
        return None

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


def smooth_max(x: Tensor, tau: float = 50.0) -> Tensor:
    """Differentiable max over all non-batch dims: (1/tau) * logsumexp(tau * x)."""
    return torch.logsumexp(tau * x.flatten(1), dim=1) / tau


def combine_scores(score_mode: str, global_score: Tensor | None, amap: Tensor) -> Tensor:
    """Map (global score, anomaly map) -> final image score according to score_mode."""
    if score_mode == "global" and global_score is not None:
        return global_score
    if score_mode == "map_max" or global_score is None and score_mode != "combined_smooth":
        return amap.flatten(1).amax(dim=1)
    if score_mode == "combined":
        return 0.5 * (global_score + amap.flatten(1).amax(dim=1))
    if score_mode == "combined_smooth":
        sm = smooth_max(amap)
        return sm if global_score is None else 0.5 * (global_score + sm)
    raise ValueError(f"Unknown score_mode '{score_mode}'")


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Differentiable 2D Gaussian blur on [B, H, W]."""
    if sigma <= 0:
        return x
    radius = int(round(3 * sigma))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = k / k.sum()
    y = x[:, None]
    y = nn.functional.pad(y, (radius, radius, radius, radius), mode="reflect")
    y = nn.functional.conv2d(y, k.view(1, 1, 1, -1))
    y = nn.functional.conv2d(y, k.view(1, 1, -1, 1))
    return y[:, 0]
