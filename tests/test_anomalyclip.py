import pytest

from src.detectors.anomalyclip import AnomalyCLIPDetector


@pytest.mark.parametrize("mode", ["combined", "combined_smooth", "map_max"])
def test_anomalyclip_shapes_and_grad(tiny_clip, images, mode):
    model, tok = tiny_clip
    det = AnomalyCLIPDetector(pretrained=None, device="cpu", score_mode=mode,
                              model=model, tokenizer=tok)
    x = images.clone().requires_grad_(True)
    score, amap = det(x)
    assert score.shape == (2,) and amap.shape == (2, 240, 240)
    assert ((amap >= 0) & (amap <= 1)).all()
    score.sum().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
