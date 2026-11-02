"""AnomalyCLIP-style detector: object-agnostic prompts + dense ViT patch tokens."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.detectors.base import BaseDetector, combine_scores, gaussian_blur
from src.detectors.winclip import load_clip

NORMAL_PROMPTS = [
    "a photo of a flawless object.",
    "a photo of a normal object without defect.",
]
ANOMALOUS_PROMPTS = [
    "a photo of a damaged object with a defect.",
    "a photo of a broken anomalous object.",
]


class AnomalyCLIPDetector(BaseDetector):
    def __init__(
        self,
        category: str | None = None,
        model_name: str = "ViT-B-16-plus-240",
        pretrained: str | None = "laion400m_e32",
        device: torch.device | str = "cpu",
        score_mode: str = "combined",
        temperature: float = 100.0,
        smooth_sigma: float = 4.0,
        learnable_offsets: bool = False,
        model=None,
        tokenizer=None,
    ) -> None:
        super().__init__(score_mode=score_mode, device=device)
        self.category = category
        if model is None:
            model, tokenizer = load_clip(model_name, pretrained, self.device)
        self.model = model
        self.temperature = temperature
        self.smooth_sigma = smooth_sigma
        with torch.no_grad():
            rows = []
            for prompts in (NORMAL_PROMPTS, ANOMALOUS_PROMPTS):
                e = F.normalize(self.model.encode_text(tokenizer(prompts).to(self.device)).float(), dim=-1)
                rows.append(F.normalize(e.mean(0), dim=-1))
            self.register_buffer("base_text", torch.stack(rows))
        # Delta T: zero-initialised prompt offsets (trainable if learnable_offsets=True).
        self.text_offset = nn.Parameter(torch.zeros_like(self.base_text),
                                        requires_grad=learnable_offsets)

    @property
    def text_features(self) -> Tensor:
        return F.normalize(self.base_text + self.text_offset, dim=-1)

    def _encode_dense(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns unit-norm pooled [B, D] and patch tokens [B, N, D] in text space."""
        v = self.model.visual
        isz = v.image_size[0] if isinstance(v.image_size, (tuple, list)) else int(v.image_size)
        if x.shape[-1] != isz:
            x = F.interpolate(x, size=(isz, isz), mode="bilinear", align_corners=False)
        h = v.transformer(v._embeds(x))
        pooled, tokens = v._pool(h)
        if v.proj is not None:
            pooled, tokens = pooled @ v.proj, tokens @ v.proj
        return F.normalize(pooled.float(), dim=-1), F.normalize(tokens.float(), dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled, tokens = self._encode_dense(x)
        t = self.text_features
        s_global = (self.temperature * pooled @ t.t()).softmax(-1)[:, 1]
        probs = (self.temperature * tokens @ t.t()).softmax(-1)[..., 1]  # [B, N]
        b, n = probs.shape
        g = int(round(n ** 0.5))
        amap = F.interpolate(probs.view(b, 1, g, g), size=x.shape[-2:], mode="bilinear",
                             align_corners=False)[:, 0]
        amap = gaussian_blur(amap, self.smooth_sigma)
        return combine_scores(self.score_mode, s_global, amap), amap
