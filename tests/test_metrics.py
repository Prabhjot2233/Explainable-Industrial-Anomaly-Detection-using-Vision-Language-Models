import numpy as np
import torch

from src.metrics.detection import image_auroc, pixel_auroc, pro_score


def test_image_auroc_perfect_and_inverse():
    assert image_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert image_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_pixel_auroc_and_pro_perfect():
    masks = np.zeros((2, 32, 32), dtype=np.float32)
    masks[0, 5:10, 5:10] = 1
    maps = masks + 0.01 * np.random.default_rng(0).random(masks.shape)
    assert pixel_auroc(masks, maps) == 1.0
    assert pro_score(masks, maps) > 0.95


def test_metrics_accept_tensors():
    assert image_auroc(torch.tensor([0, 1]), torch.tensor([0.0, 1.0])) == 1.0
