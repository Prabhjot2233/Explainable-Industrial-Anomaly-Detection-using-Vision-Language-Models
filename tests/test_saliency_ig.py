import pytest
import torch

from src.explainers.base import DifferentiableScoreWrapper
from src.explainers.ig import IntegratedGradientsExplainer
from src.explainers.saliency import SaliencyExplainer


def _check_map(out):
    m = out["attribution_map"]
    assert m.shape == (1, 240, 240)
    assert m.isfinite().all()
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.sum() > 0


# ---------------- Part A: wrapper + saliency
def test_wrapper_returns_1d_and_restores_mode(make_winclip, images):
    det = make_winclip(score_mode="combined")
    w = DifferentiableScoreWrapper(det)
    out = w(images)
    assert out.shape == (2,)
    assert det.score_mode == "combined"


def test_saliency_explainer(make_winclip, images):
    det = make_winclip()
    out = SaliencyExplainer(det, torch.device("cpu")).explain(images[:1])
    _check_map(out)
    assert out["raw_attribution"].shape == (1, 3, 240, 240)
    assert det.score_mode == "combined"


# ---------------- Part B: integrated gradients
def test_ig_black_completeness(make_winclip, images):
    det = make_winclip()
    out = IntegratedGradientsExplainer(det, "cpu", baseline_mode="black", n_steps=8).explain(images[:1])
    _check_map(out)
    assert out["meta"]["completeness_delta"] < 0.05


def test_ig_expected_differs_from_black(make_winclip, images):
    det = make_winclip()
    img = images[:1]
    refs = torch.randn(2, 3, 240, 240, generator=torch.Generator().manual_seed(5)) * 0.3
    black = IntegratedGradientsExplainer(det, "cpu", "black", n_steps=6).explain(img)
    exp = IntegratedGradientsExplainer(det, "cpu", "expected", n_steps=6).explain(img, refs)
    _check_map(exp)
    assert not torch.allclose(black["attribution_map"], exp["attribution_map"])


def test_ig_expected_requires_refs(make_winclip, images):
    with pytest.raises(ValueError):
        IntegratedGradientsExplainer(make_winclip(), "cpu", "expected").explain(images[:1])
