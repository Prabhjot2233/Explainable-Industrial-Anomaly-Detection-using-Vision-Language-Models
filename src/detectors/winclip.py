"""WinCLIP / WinCLIP+ detector over OpenCLIP ViT-B-16-plus-240.

Stage A: global image vs. text prompts -> s_global [B].
Stage B: sliding windows (scale s -> s*patch px) encoded by CLIP, scored vs. text,
         aggregated per pixel with a clamped harmonic mean -> M_text [B,240,240].
Few-shot (WinCLIP+): test window embeddings are compared to stored normal-reference
window embeddings -> M_vis; fused as (1-a) M_text + a M_vis.
"""
from __future__ import annotations

from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import Tensor

from src.detectors.base import BaseDetector, combine_scores, gaussian_blur
from src.detectors.prompts import build_text_embedding


def load_clip(model_name: str, pretrained: str | None, device: torch.device):
    model = open_clip.create_model(model_name, pretrained=pretrained or None, device=device)
    model = model.float().eval()
    model.requires_grad_(False)
    try:
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception:  # unknown/custom configs fall back to the default CLIP BPE tokenizer
        tokenizer = open_clip.tokenize
    return model, tokenizer


class WinCLIPDetector(BaseDetector):
    def __init__(
        self,
        category: str,
        model_name: str = "ViT-B-16-plus-240",
        pretrained: str | None = "laion400m_e32",
        device: torch.device | str = "cpu",
        image_size: int = 240,
        window_scales: list[int] | tuple[int, ...] = (2, 3),
        window_stride: int | None = None,
        window_chunk_size: int = 16,
        score_mode: str = "combined",
        prompt_mode: str = "full_ensemble",
        temperature: float = 100.0,
        alpha_vis: float = 0.5,
        tau_ref: float = 50.0,
        smooth_sigma: float = 0.0,
        cache_dir: str | Path | None = "results/cache",
        model=None,
        tokenizer=None,
    ) -> None:
        super().__init__(score_mode=score_mode, device=device)
        if model is None:
            model, tokenizer = load_clip(model_name, pretrained, self.device)
        self.model = model
        self.category = category
        self.image_size = image_size
        self.window_scales = list(window_scales)
        ps = self.model.visual.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        self.window_stride = window_stride or self.patch_size
        self.window_chunk_size = window_chunk_size
        self.temperature = temperature
        self.alpha_vis = alpha_vis
        self.tau_ref = tau_ref
        self.smooth_sigma = smooth_sigma
        isz = self.model.visual.image_size
        self.clip_input_size = isz[0] if isinstance(isz, (tuple, list)) else int(isz)
        self.prompt_mode = prompt_mode
        self.text_features = build_text_embedding(
            self.model, tokenizer, category, self.device, prompt_mode=prompt_mode,
            cache_dir=cache_dir, pretrained_tag=pretrained,
        ).to(self.device)
        self.ref_window_features: dict[int, Tensor] = {}
        self.is_fewshot = False

    # ------------------------------------------------------------------ helpers
    def _encode(self, imgs: Tensor) -> Tensor:
        """Encode images [N,3,h,w] (resized to CLIP input size) -> unit-norm [N, D]."""
        if imgs.shape[-1] != self.clip_input_size or imgs.shape[-2] != self.clip_input_size:
            imgs = F.interpolate(imgs, size=(self.clip_input_size,) * 2, mode="bilinear",
                                 align_corners=False)
        return F.normalize(self.model.encode_image(imgs).float(), dim=-1)

    def _window_size(self, scale: int) -> int:
        return scale * self.patch_size

    def _encode_windows(self, x: Tensor, scale: int) -> Tensor:
        """x [B,3,H,W] -> unit-norm window embeddings [B, N_win, D]."""
        b = x.shape[0]
        w = self._window_size(scale)
        cols = F.unfold(x, kernel_size=w, stride=self.window_stride)  # [B, 3*w*w, N]
        n = cols.shape[-1]
        wins = cols.transpose(1, 2).reshape(b * n, 3, w, w)
        # Use a conservative chunk size (8) during gradient calculation to avoid MPS memory spikes
        chunk_size = min(self.window_chunk_size, 8) if torch.is_grad_enabled() else self.window_chunk_size
        feats = [self._encode(wins[i:i + chunk_size])
                 for i in range(0, wins.shape[0], chunk_size)]
        return torch.cat(feats, dim=0).view(b, n, -1)

    def _text_prob(self, feats: Tensor) -> Tensor:
        """Anomalous-class probability for unit-norm features [..., D]."""
        logits = self.temperature * feats @ self.text_features.t()
        return logits.softmax(dim=-1)[..., 1]

    def _fold_harmonic(self, win_scores: Tensor, scale: int, hw: tuple[int, int]) -> Tensor:
        """Per-pixel harmonic mean of the window scores covering it -> [B, H, W]."""
        w = self._window_size(scale)
        inv = 1.0 / win_scores.clamp(min=1e-6)  # [B, N]
        patches = inv[:, None, :].expand(-1, w * w, -1)
        sum_inv = F.fold(patches, output_size=hw, kernel_size=w, stride=self.window_stride)
        count = F.fold(torch.ones_like(patches), output_size=hw, kernel_size=w,
                       stride=self.window_stride)
        amap = torch.where(count > 0, count / sum_inv.clamp(min=1e-6), torch.zeros_like(count))
        return amap[:, 0]

    # ------------------------------------------------------------------ few-shot
    def set_reference(self, normal_images: Tensor | None) -> None:
        if normal_images is None:
            self.ref_window_features = {}
            self.is_fewshot = False
            return
        normal_images = normal_images.to(self.device)
        with torch.no_grad():
            for s in self.window_scales:
                f = self._encode_windows(normal_images, s)  # [K, N, D]
                self.ref_window_features[s] = f.reshape(-1, f.shape[-1]).detach()
        self.is_fewshot = True

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hw = (x.shape[-2], x.shape[-1])
        # Stage A: global (encoded first so Grad-CAM hooks see the full image first)
        s_global = self._text_prob(self._encode(x))

        text_maps, vis_maps = [], []
        for s in self.window_scales:
            fwin = self._encode_windows(x, s)  # Stage B, encoded once
            text_maps.append(self._fold_harmonic(self._text_prob(fwin), s, hw))
            if self.is_fewshot:
                sim = fwin @ self.ref_window_features[s].t()  # [B, N, M]
                if self.score_mode == "combined_smooth":
                    nn_sim = torch.logsumexp(self.tau_ref * sim, dim=-1) / self.tau_ref
                else:
                    nn_sim = sim.amax(dim=-1)
                s_vis = (0.5 * (1.0 - nn_sim)).clamp(min=0.0)
                vis_maps.append(self._fold_harmonic(s_vis, s, hw))

        amap = torch.stack(text_maps).mean(dim=0)
        if self.is_fewshot:
            m_vis = torch.stack(vis_maps).mean(dim=0)
            amap = (1 - self.alpha_vis) * amap + self.alpha_vis * m_vis
        amap = gaussian_blur(amap, self.smooth_sigma)
        return combine_scores(self.score_mode, s_global, amap), amap
