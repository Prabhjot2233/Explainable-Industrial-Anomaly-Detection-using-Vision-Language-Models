import pytest
import numpy as np
import torch
from torch import Tensor
from src.detectors.base import BaseDetector
from src.metrics.explanation import (
    correctness_auc,
    correctness_sum,
    consistency_del,
    consistency_add,
)


class DummyRegionDetector(BaseDetector):
    """Simple dummy detector where score is mean pixel intensity of a designated region."""
    def __init__(self, score_mode: str = "combined"):
        super().__init__(score_mode=score_mode, device="cpu")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x is [B, 3, H, W]
        # Region of interest is [8:14, 8:14]
        roi = x[:, :, 8:14, 8:14]
        score = roi.mean(dim=(1, 2, 3))
        amap = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        amap[:, 8:14, 8:14] = 1.0
        return score, amap


def test_correctness_auc_and_sum():
    gt_mask = np.zeros((20, 20), dtype=np.float32)
    gt_mask[5:15, 5:15] = 1.0

    # Perfect attribution
    perf_attr = gt_mask.copy()
    assert correctness_auc(perf_attr, gt_mask) == 1.0
    assert pytest.approx(correctness_sum(perf_attr, gt_mask), abs=1e-5) == 1.0

    # Inverted attribution
    inv_attr = 1.0 - gt_mask
    assert correctness_auc(inv_attr, gt_mask) == 0.0
    assert pytest.approx(correctness_sum(inv_attr, gt_mask), abs=1e-5) == 0.0

    # Constant attribution
    const_attr = np.ones((20, 20), dtype=np.float32)
    assert correctness_auc(const_attr, gt_mask) == 0.5


def test_consistency_del_and_add():
    det = DummyRegionDetector()
    
    # Create normal image (all zeros) and anomalous image (roi filled with 2.0)
    normal_img = torch.zeros(1, 3, 20, 20)
    anom_img = torch.zeros(1, 3, 20, 20)
    anom_img[:, :, 8:14, 8:14] = 2.0
    
    # Ground-truth concentrated attribution map
    attr_map = torch.zeros(1, 20, 20)
    attr_map[:, 8:14, 8:14] = 1.0
    
    del_res = consistency_del(det, anom_img, attr_map, baseline_image=normal_img, steps=5)
    assert "audc" in del_res
    assert len(del_res["fractions"]) == 6
    assert len(del_res["scores"]) == 6
    # Score should drop as we delete
    assert del_res["scores"][0] > del_res["scores"][-1]

    add_res = consistency_add(det, normal_img, anom_img, attr_map, steps=5)
    assert "auac" in add_res
    assert len(add_res["fractions"]) == 6
    assert len(add_res["scores"]) == 6
    # Score should rise as we add
    assert add_res["scores"][0] < add_res["scores"][-1]
    
    # Faithfulness: AUAC should be significantly higher than AUDC
    assert add_res["auac"] > del_res["audc"]
