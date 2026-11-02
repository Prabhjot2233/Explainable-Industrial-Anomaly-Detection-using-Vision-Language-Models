"""Vanilla input-gradient saliency via Captum."""
from __future__ import annotations

import time

from captum.attr import Saliency
from torch import Tensor

from src.explainers.base import BaseExplainer


class SaliencyExplainer(BaseExplainer):
    def explain(self, image: Tensor, reference_normals: Tensor | None = None) -> dict:
        t0 = time.perf_counter()
        image = image.detach().clone().to(self.device).requires_grad_(True)
        raw_attr = Saliency(self.score_wrapper).attribute(image, abs=False).detach()
        return {
            "attribution_map": self.aggregate_and_normalize(raw_attr).float(),
            "raw_attribution": raw_attr.float(),
            "baseline": None,
            "meta": {"method": "saliency", "seconds": time.perf_counter() - t0},
        }
