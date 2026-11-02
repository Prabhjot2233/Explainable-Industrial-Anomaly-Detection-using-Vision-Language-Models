import pytest
import torch

from src.detectors.patchcore import PatchCoreDetector
from src.utils.device import get_device


@pytest.fixture(scope="module")
def patchcore():
    # pretrained=False keeps the test offline; architecture & math are identical.
    torch.manual_seed(0)
    return PatchCoreDetector(device=get_device("cpu"), pretrained=False, coreset_ratio=1.0)


def _smooth_images(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 30, 30, generator=g)
    return torch.nn.functional.interpolate(x, size=(240, 240), mode="bilinear") * 0.5


def test_memory_bank(patchcore):
    patchcore.set_reference(_smooth_images(2))
    assert patchcore.memory_bank is not None
    assert patchcore.memory_bank.shape == (2 * 900, 1536)


def test_forward_shapes_and_grad(patchcore):
    patchcore.set_reference(_smooth_images(2))
    x = _smooth_images(2, seed=1).requires_grad_(True)
    score, amap = patchcore(x)
    assert score.shape == (2,) and amap.shape == (2, 240, 240)
    score.sum().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0


def test_defect_scores_higher(patchcore):
    normals = _smooth_images(2)
    patchcore.set_reference(normals)
    defect = normals[:1].clone()
    defect[:, :, 100:140, 100:140] = 3.0  # high-contrast square
    with torch.no_grad():
        s, _ = patchcore(torch.cat([normals[:1], defect]))
    assert s[1] > s[0]
