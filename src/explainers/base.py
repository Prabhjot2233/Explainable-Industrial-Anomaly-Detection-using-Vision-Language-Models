"""BaseExplainer contract and the scalar-score wrapper used by Captum."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.detectors.base import BaseDetector


class DifferentiableScoreWrapper(nn.Module):
    """Wraps a BaseDetector so forward(x) returns ONLY a 1D tensor of scalar scores [B].
    Automatically switches detector score_mode to 'combined_smooth' during attribution
    and restores the original score_mode afterward.
    """

    def __init__(self, detector: BaseDetector, target_score_mode: str = "combined_smooth"):
        super().__init__()
        self.detector = detector
        self.target_score_mode = target_score_mode

    def forward(self, x: Tensor) -> Tensor:
        old_mode = getattr(self.detector, "score_mode", None)
        try:
            if old_mode is not None:
                self.detector.score_mode = self.target_score_mode
            scores, _ = self.detector(x)
            return scores
        finally:
            if old_mode is not None:
                self.detector.score_mode = old_mode


class BaseExplainer(ABC):
    def __init__(self, detector: BaseDetector, device: torch.device):
        self.detector = detector
        self.device = torch.device(device)
        self.score_wrapper = DifferentiableScoreWrapper(detector)

    @staticmethod
    def aggregate_and_normalize(raw_attr: Tensor, eps: float = 1e-8) -> Tensor:
        """Converts raw attribution [B, 3, H, W] -> [B, H, W] in [0.0, 1.0].
        Takes L2 norm across color channels (dim=1), then min-max normalizes per image.
        """
        spatial = torch.linalg.vector_norm(raw_attr, ord=2, dim=1)  # [B, H, W]
        return BaseExplainer.minmax(spatial, eps)

    @staticmethod
    def minmax(spatial: Tensor, eps: float = 1e-8) -> Tensor:
        b = spatial.shape[0]
        flat = spatial.reshape(b, -1)
        mins = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1)
        maxs = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1)
        return (spatial - mins) / (maxs - mins + eps)

    @abstractmethod
    def explain(
        self,
        image: Tensor,                            # [1, 3, 240, 240]
        reference_normals: Tensor | None = None,  # Optional [K, 3, 240, 240]
    ) -> dict[str, Tensor | dict]:
        """Returns:
        {
            "attribution_map": Tensor [1, 240, 240] float32 in [0, 1],
            "raw_attribution": Tensor [1, 3, 240, 240] float32,
            "baseline": Tensor [1, 3, 240, 240] float32 (or None),
            "meta": dict
        }
        """
        raise NotImplementedError
