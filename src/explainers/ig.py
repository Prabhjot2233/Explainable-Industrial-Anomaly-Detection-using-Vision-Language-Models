"""Integrated Gradients with black baseline or Expected-IG over normal references."""
from __future__ import annotations

import time

import torch
from captum.attr import IntegratedGradients
from torch import Tensor

from src.data.mvtec import clip_normalize
from src.explainers.base import BaseExplainer

VALID_BASELINES = ("black", "true_black", "expected")


class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, detector, device, baseline_mode: str = "black", n_steps: int = 20,
                 internal_batch_size: int = 4):
        super().__init__(detector, device)
        if baseline_mode not in VALID_BASELINES:
            raise ValueError(f"baseline_mode must be one of {VALID_BASELINES}")
        self.baseline_mode = baseline_mode
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def _score(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.score_wrapper(x).sum()

    def explain(self, image: Tensor, reference_normals: Tensor | None = None) -> dict:
        t0 = time.perf_counter()
        image = image.detach().clone().to(self.device).requires_grad_(True)
        ig = IntegratedGradients(self.score_wrapper)
        meta: dict = {"method": f"ig_{self.baseline_mode}", "n_steps": self.n_steps}

        if self.baseline_mode == "expected":
            if reference_normals is None:
                raise ValueError("baseline_mode='expected' requires reference_normals [K,3,H,W].")
            refs = reference_normals.detach().to(self.device)
            k = refs.shape[0]
            steps = max(10, self.n_steps // k)
            attrs = [
                ig.attribute(image, baselines=refs[i:i + 1], n_steps=steps,
                             internal_batch_size=self.internal_batch_size).detach()
                for i in range(k)
            ]
            raw_attr = torch.stack(attrs).mean(dim=0)
            baseline = refs.mean(dim=0, keepdim=True)
            meta.update(k_baselines=k, steps_per_baseline=steps)
        else:
            if self.baseline_mode == "black":
                baseline = torch.zeros_like(image)  # mean CLIP colour in normalised space
            else:  # true_black: RGB (0,0,0) after CLIP normalisation
                baseline = clip_normalize(torch.zeros_like(image[0]))[None]
            raw_attr = ig.attribute(image, baselines=baseline.detach(), n_steps=self.n_steps,
                                    internal_batch_size=self.internal_batch_size).detach()
            delta = abs(raw_attr.sum().item()
                        - (self._score(image) - self._score(baseline)).item())
            meta["completeness_delta"] = delta

        meta["seconds"] = time.perf_counter() - t0
        return {
            "attribution_map": self.aggregate_and_normalize(raw_attr).float(),
            "raw_attribution": raw_attr.float(),
            "baseline": baseline.detach().float(),
            "meta": meta,
        }
