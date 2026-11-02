import numpy as np
import pytest
from src.metrics.language_eval import (
    compute_defect_type_accuracy,
    compute_bbox_iou,
    compute_mask_quadrant_and_bbox,
    compute_region_grounding,
    compute_hallucination_rate,
)


def test_defect_type_accuracy():
    assert compute_defect_type_accuracy("crack", "broken_large") is True
    assert compute_defect_type_accuracy("hole", "poke") is True
    assert compute_defect_type_accuracy("none", "good") is True
    assert compute_defect_type_accuracy("scratch", "broken_large") is False


def test_bbox_iou():
    boxA = (10, 10, 50, 50)
    # Identical
    assert compute_bbox_iou(boxA, boxA) == 1.0

    # No overlap
    boxB = (60, 60, 100, 100)
    assert compute_bbox_iou(boxA, boxB) == 0.0

    # Partial overlap
    boxC = (30, 30, 50, 50)  # Area 400, intersection with boxA (area 1600) is 400
    assert 0.0 < compute_bbox_iou(boxA, boxC) < 1.0


def test_region_grounding():
    mask = np.zeros((240, 240), dtype=np.float32)
    mask[20:60, 20:60] = 1.0

    res = compute_region_grounding(
        pred_bbox=(20, 20, 60, 60),
        pred_quadrant="top-left",
        gt_mask=mask,
    )
    assert res["quadrant_match"] is True
    assert res["bbox_iou"] == 1.0
    assert res["gt_quadrant"] == "top-left"


def test_hallucination_rate():
    preds = [
        {"is_gt_normal": True, "predicted_is_anomalous": False},  # Correct normal
        {"is_gt_normal": True, "predicted_is_anomalous": True},   # Hallucinated defect
        {"is_gt_normal": False, "bbox_iou": 0.45},               # Grounded defect
        {"is_gt_normal": False, "bbox_iou": 0.0},                # Spatial hallucination
    ]
    rates = compute_hallucination_rate(preds)
    assert rates["normal_false_positive_rate"] == 0.5
    assert rates["spatial_hallucination_rate"] == 0.5
    assert rates["overall_hallucination_score"] == 0.5

