from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import scipy.ndimage as ndimage
import torch


DEFECT_FAMILY_MAP = {
    # MVTec carpet defects
    "cut": "cut",
    "hole": "hole",
    "thread": "contamination",
    "metal_contamination": "contamination",
    "color": "contamination",
    # MVTec bottle defects
    "broken_large": "crack",
    "broken_small": "crack",
    "contamination": "contamination",
    # MVTec leather defects
    "fold": "deformation",
    "glue": "contamination",
    "poke": "hole",
    "scratch": "scratch",
    # MVTec tile defects
    "crack": "crack",
    "gray_stroke": "contamination",
    "oil": "contamination",
    "rough": "scratch",
    # MVTec transistor defects
    "bent_lead": "deformation",
    "cut_lead": "cut",
    "damaged_case": "crack",
    "misplaced": "deformation",
    "good": "none",
}


def compute_defect_type_accuracy(
    predicted_defect: str,
    ground_truth_defect: str,
) -> bool:
    """Checks whether predicted defect matches ground-truth defect family."""
    pred = predicted_defect.strip().lower()
    gt = ground_truth_defect.strip().lower()

    if pred == "none" and gt in ["good", "none", "normal"]:
        return True

    # Exact or substring match
    if pred == gt or pred in gt or gt in pred:
        return True

    # Map to canonical defect families
    expected_family = DEFECT_FAMILY_MAP.get(gt, gt)
    return pred == expected_family


def compute_bbox_iou(
    bbox_pred: Tuple[int, int, int, int],
    bbox_gt: Tuple[int, int, int, int],
) -> float:
    """Computes Intersection over Union between two bounding boxes (x_min, y_min, x_max, y_max)."""
    px0, py0, px1, py1 = bbox_pred
    gx0, gy0, gx1, gy1 = bbox_gt

    if px1 <= px0 or py1 <= py0 or gx1 <= gx0 or gy1 <= gy0:
        return 0.0

    inter_x0 = max(px0, gx0)
    inter_y0 = max(py0, gy0)
    inter_x1 = min(px1, gx1)
    inter_y1 = min(py1, gy1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    p_area = (px1 - px0) * (py1 - py0)
    g_area = (gx1 - gx0) * (gy1 - gy0)
    union_area = p_area + g_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def compute_mask_quadrant_and_bbox(
    gt_mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], str]:
    """Finds bounding box and quadrant of ground-truth defect mask."""
    mask_bin = (gt_mask.squeeze() > 0.5).astype(np.uint8)
    if not mask_bin.any():
        return (0, 0, 0, 0), "none"

    coords = np.argwhere(mask_bin)
    y_min, x_min = int(coords[:, 0].min()), int(coords[:, 1].min())
    y_max, x_max = int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1

    h, w = mask_bin.shape[-2:]
    cy, cx = ndimage.center_of_mass(mask_bin)

    if (h * 0.33) <= cy <= (h * 0.67) and (w * 0.33) <= cx <= (w * 0.67):
        quadrant = "center"
    else:
        v_str = "top" if cy < (h / 2.0) else "bottom"
        h_str = "left" if cx < (w / 2.0) else "right"
        quadrant = f"{v_str}-{h_str}"

    return (x_min, y_min, x_max, y_max), quadrant


def compute_region_grounding(
    pred_bbox: Tuple[int, int, int, int],
    pred_quadrant: str,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluates spatial grounding: Quadrant Match and Bounding-Box IoU."""
    gt_bbox, gt_quadrant = compute_mask_quadrant_and_bbox(gt_mask)
    if gt_quadrant == "none":
        quad_match = pred_quadrant == "none"
        iou = 1.0 if pred_bbox == (0, 0, 0, 0) else 0.0
    else:
        quad_match = pred_quadrant == gt_quadrant
        iou = compute_bbox_iou(pred_bbox, gt_bbox)

    return {
        "quadrant_match": bool(quad_match),
        "bbox_iou": float(iou),
        "gt_bbox": gt_bbox,
        "gt_quadrant": gt_quadrant,
    }


def compute_hallucination_rate(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """Computes hallucination rate on normal and anomalous samples.
    - normal_false_positive_rate: Claiming defect on a normal image.
    - spatial_hallucination_rate: Claiming a location with zero IoU on an anomalous image.
    """
    normal_claims = 0
    normal_total = 0

    spatial_hallucinations = 0
    anomalous_total = 0

    for p in predictions:
        is_gt_normal = p.get("is_gt_normal", False)
        if is_gt_normal:
            normal_total += 1
            if p.get("predicted_is_anomalous", False):
                normal_claims += 1
        else:
            anomalous_total += 1
            iou = p.get("bbox_iou", 0.0)
            if iou <= 0.0:
                spatial_hallucinations += 1

    norm_fpr = (normal_claims / normal_total) if normal_total > 0 else 0.0
    spat_hr = (spatial_hallucinations / anomalous_total) if anomalous_total > 0 else 0.0

    return {
        "normal_false_positive_rate": float(norm_fpr),
        "spatial_hallucination_rate": float(spat_hr),
        "overall_hallucination_score": float(0.5 * norm_fpr + 0.5 * spat_hr),
    }
