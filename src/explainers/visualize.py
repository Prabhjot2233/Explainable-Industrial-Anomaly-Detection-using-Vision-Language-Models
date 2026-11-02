"""Heatmap overlays and multi-panel comparison figures (matplotlib Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from torch import Tensor  # noqa: E402

from src.data.mvtec import clip_denormalize  # noqa: E402

PANEL_ORDER = ["Saliency", "ViT Grad-CAM", "IG (Black)", "Expected-IG (Normal)"]


def _to_np2d(t) -> np.ndarray:
    if isinstance(t, Tensor):
        t = t.detach().float().cpu().numpy()
    return np.squeeze(np.asarray(t, dtype=np.float32))


def denormalize_clip_image(img_tensor: Tensor) -> np.ndarray:
    """[3,H,W] CLIP-normalised -> [H,W,3] float32 RGB in [0,1]."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    rgb = clip_denormalize(img_tensor.detach().float().cpu()).clamp(0, 1)
    return rgb.permute(1, 2, 0).numpy().astype(np.float32)


def create_heatmap_overlay(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45,
                           cmap: str = "jet") -> np.ndarray:
    heat = np.clip(_to_np2d(heatmap), 0.0, 1.0)
    colored = colormaps[cmap](heat)[..., :3].astype(np.float32)
    return np.clip((1 - alpha) * rgb_img + alpha * colored, 0.0, 1.0)


def save_explanation_comparison_grid(
    image: Tensor,
    gt_mask: Tensor,
    anomaly_map: Tensor,
    attributions: dict[str, Tensor],
    save_path: str | Path,
    title: str = "",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = denormalize_clip_image(image)
    amap = _to_np2d(anomaly_map)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    names = [n for n in PANEL_ORDER if n in attributions]
    names += [n for n in attributions if n not in names]
    panels = [("Original RGB", rgb), ("Ground Truth Mask", _to_np2d(gt_mask)),
              ("Detector Anomaly Map", create_heatmap_overlay(rgb, amap))]
    panels += [(n, create_heatmap_overlay(rgb, _to_np2d(attributions[n]))) for n in names]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0))
    for ax, (name, arr) in zip(np.atleast_1d(axes), panels):
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
