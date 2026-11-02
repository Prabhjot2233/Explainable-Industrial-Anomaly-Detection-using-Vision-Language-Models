"""Grad-CAM for OpenCLIP Vision Transformers (last residual block patch tokens)."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor

from src.explainers.base import BaseExplainer


class ViTGradCAMExplainer(BaseExplainer):
    """Hooks detector.model.visual.transformer.resblocks[-1].

    Detectors such as WinCLIP call the visual encoder many times per forward (global image +
    every window). Only the FIRST call that matches the input batch size (the full-image
    pass) is captured; its gradient is grabbed with a tensor hook registered on that output.
    """

    def _target_block(self):
        return self.detector.model.visual.transformer.resblocks[-1]

    def explain(self, image: Tensor, reference_normals: Tensor | None = None) -> dict:
        t0 = time.perf_counter()
        image = image.detach().clone().to(self.device).requires_grad_(True)
        transformer = self.detector.model.visual.transformer
        batch_first = getattr(transformer, "batch_first", True)
        block = self._target_block()
        store: dict[str, Tensor] = {}

        def fwd_hook(_module, _inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            bt = out_t if batch_first else out_t.transpose(0, 1)
            if "act" in store or bt.shape[0] != image.shape[0]:
                return
            store["act"] = bt.detach()

            def grad_hook(g):
                store["grad"] = (g if batch_first else g.transpose(0, 1)).detach()

            out_t.register_hook(grad_hook)

        handle = block.register_forward_hook(fwd_hook)
        try:
            score = self.score_wrapper(image)
            self.detector.zero_grad(set_to_none=True)
            score.sum().backward()
        finally:
            handle.remove()

        if "act" not in store or "grad" not in store:
            raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")
        a, g = store["act"][:, 1:], store["grad"][:, 1:]  # drop [CLS]
        n = a.shape[1]
        side = int(round(n ** 0.5))
        w = g.mean(dim=1, keepdim=True)                        # [1, 1, D]
        cam = F.relu((w * a).sum(dim=-1)).view(1, 1, side, side)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bicubic", align_corners=False)
        cam = self.minmax(cam[:, 0].clamp(min=0)).float()
        return {
            "attribution_map": cam,
            "raw_attribution": cam[:, None].expand(-1, 3, -1, -1).clone(),
            "baseline": None,
            "meta": {"method": "vit_gradcam", "grid": side, "seconds": time.perf_counter() - t0},
        }
