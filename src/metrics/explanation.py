from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor

from src.detectors.base import BaseDetector


def correctness_auc(attr_map: np.ndarray | Tensor, gt_mask: np.ndarray | Tensor) -> float:
    """Computes pixel-level ROC-AUC between an attribution map and ground-truth defect mask."""
    if isinstance(attr_map, Tensor):
        attr_map = attr_map.detach().cpu().numpy()
    if isinstance(gt_mask, Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    flat_attr = np.asarray(attr_map, dtype=np.float32).ravel()
    flat_mask = (np.asarray(gt_mask, dtype=np.float32).ravel() > 0.5).astype(np.int32)

    if len(np.unique(flat_mask)) < 2:
        return float("nan")
    if np.allclose(flat_attr, flat_attr[0]):
        return 0.5

    return float(roc_auc_score(flat_mask, flat_attr))


def correctness_sum(attr_map: np.ndarray | Tensor, gt_mask: np.ndarray | Tensor) -> float:
    """Computes the fraction of total attribution mass falling inside the ground-truth mask."""
    if isinstance(attr_map, Tensor):
        attr_map = attr_map.detach().cpu().numpy()
    if isinstance(gt_mask, Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    attr = np.abs(np.asarray(attr_map, dtype=np.float32)).squeeze()
    mask = (np.asarray(gt_mask, dtype=np.float32).squeeze() > 0.5).astype(np.float32)

    total_mass = float(np.sum(attr))
    if total_mass <= 1e-8 or float(np.sum(mask)) == 0.0:
        return 0.0

    inside_mass = float(np.sum(attr * mask))
    return float(np.clip(inside_mass / (total_mass + 1e-8), 0.0, 1.0))


def consistency_del(
    detector: BaseDetector,
    image: Tensor,
    attr_map: Tensor | np.ndarray,
    baseline_image: Tensor | None = None,
    steps: int = 10,
) -> dict[str, Any]:
    """Deletion test: progressively replaces top-attributed pixels with baseline values.
    Lower AUDC (Area Under Deletion Curve) indicates a more faithful explanation.
    """
    device = next(detector.parameters()).device if list(detector.parameters()) else image.device
    x = image.to(device, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)

    if baseline_image is None:
        base = torch.zeros_like(x)
    else:
        base = baseline_image.to(device, dtype=torch.float32)
        if base.ndim == 3:
            base = base.unsqueeze(0)

    if isinstance(attr_map, np.ndarray):
        attr_t = torch.from_numpy(attr_map).to(device, dtype=torch.float32)
    else:
        attr_t = attr_map.to(device, dtype=torch.float32)
    attr_flat = attr_t.reshape(-1)
    sorted_indices = torch.argsort(attr_flat, descending=True)

    num_pixels = attr_flat.numel()
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores: list[float] = []

    with torch.no_grad():
        for frac in fractions:
            k = int(round(frac * num_pixels))
            x_mod = x.clone()
            if k > 0:
                idx = sorted_indices[:k]
                x_mod_flat = x_mod.view(1, 3, -1)
                base_flat = base.view(1, 3, -1)
                x_mod_flat[:, :, idx] = base_flat[:, :, idx]
            score_t, _ = detector(x_mod)
            scores.append(float(score_t.view(-1)[0].item()))

    audc = float(np.trapezoid(scores, fractions)) if hasattr(np, "trapezoid") else float(np.trapz(scores, fractions))
    return {
        "audc": audc,
        "fractions": [float(f) for f in fractions],
        "scores": scores,
    }


def consistency_add(
    detector: BaseDetector,
    normal_image: Tensor,
    anomalous_image: Tensor,
    attr_map: Tensor | np.ndarray,
    steps: int = 10,
) -> dict[str, Any]:
    """Addition test: progressively inserts top-attributed anomalous pixels into a normal image.
    Higher AUAC (Area Under Addition Curve) indicates a more faithful explanation.
    """
    device = (
        next(detector.parameters()).device
        if list(detector.parameters())
        else anomalous_image.device
    )
    x_norm = normal_image.to(device, dtype=torch.float32)
    if x_norm.ndim == 3:
        x_norm = x_norm.unsqueeze(0)

    x_anom = anomalous_image.to(device, dtype=torch.float32)
    if x_anom.ndim == 3:
        x_anom = x_anom.unsqueeze(0)

    if isinstance(attr_map, np.ndarray):
        attr_t = torch.from_numpy(attr_map).to(device, dtype=torch.float32)
    else:
        attr_t = attr_map.to(device, dtype=torch.float32)
    attr_flat = attr_t.reshape(-1)
    sorted_indices = torch.argsort(attr_flat, descending=True)

    num_pixels = attr_flat.numel()
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores: list[float] = []

    with torch.no_grad():
        for frac in fractions:
            k = int(round(frac * num_pixels))
            x_comp = x_norm.clone()
            if k > 0:
                idx = sorted_indices[:k]
                x_comp_flat = x_comp.view(1, 3, -1)
                x_anom_flat = x_anom.view(1, 3, -1)
                x_comp_flat[:, :, idx] = x_anom_flat[:, :, idx]
            score_t, _ = detector(x_comp)
            scores.append(float(score_t.view(-1)[0].item()))

    auac = float(np.trapezoid(scores, fractions)) if hasattr(np, "trapezoid") else float(np.trapz(scores, fractions))
    return {
        "auac": auac,
        "fractions": [float(f) for f in fractions],
        "scores": scores,
    }

