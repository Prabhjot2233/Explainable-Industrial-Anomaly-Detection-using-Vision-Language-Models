"""Detection metrics: image AUROC, pixel AUROC, PRO (per-region overlap, FPR <= 0.3)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def image_auroc(labels, scores) -> float:
    labels, scores = _np(labels).ravel(), _np(scores).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, maps) -> float:
    masks = (_np(masks).ravel() > 0.5).astype(np.uint8)
    maps = _np(maps).ravel()
    if len(np.unique(masks)) < 2:
        return float("nan")
    return float(roc_auc_score(masks, maps))


def pro_score(masks, maps, fpr_limit: float = 0.3, num_thresholds: int = 200) -> float:
    """Area under the PRO curve up to fpr_limit, normalised to [0, 1].

    masks, maps: [N, H, W].
    """
    masks = _np(masks).astype(bool)
    maps = _np(maps).astype(np.float64)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if not masks.any():
        return float("nan")

    regions = []  # (image_idx, boolean region mask)
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m)
        for r in range(1, n + 1):
            regions.append((i, lab == r))
    neg = ~masks
    n_neg = neg.sum()

    thresholds = np.linspace(maps.max(), maps.min(), num_thresholds)
    fprs, pros = [], []
    for t in thresholds:
        pred = maps >= t
        fpr = (pred & neg).sum() / max(n_neg, 1)
        pro = np.mean([pred[i][r].mean() for i, r in regions])
        fprs.append(fpr)
        pros.append(pro)
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    if keep.sum() < 2:
        return 0.0
    f, p = fprs[keep], pros[keep]
    # Close the curve at exactly fpr_limit by interpolation.
    if f[-1] < fpr_limit and (~keep).any():
        j = np.argmax(~keep)
        f2, p2 = fprs[j], pros[j]
        p_end = p[-1] + (p2 - p[-1]) * (fpr_limit - f[-1]) / max(f2 - f[-1], 1e-12)
        f, p = np.append(f, fpr_limit), np.append(p, p_end)
    return float(np.trapezoid(p, f) / fpr_limit) if hasattr(np, "trapezoid") else float(np.trapz(p, f) / fpr_limit)
