import torch
import pytest
from src.explainers.abigx import ABIGXExplainer


def test_abigx_explainer_basic(make_winclip, images):
    det = make_winclip()
    device = torch.device("cpu")
    explainer = ABIGXExplainer(
        detector=det,
        device=device,
        afr_steps=3,
        afr_step_size=0.03,
        ig_steps=4,
    )
    
    img = images[:1]
    res = explainer.explain(img)
    
    attr_map = res["attribution_map"]
    assert attr_map.shape == (1, 240, 240)
    assert float(attr_map.min()) >= 0.0
    assert float(attr_map.max()) <= 1.0 + 1e-6
    assert res["meta"]["baseline_score"] <= res["meta"]["initial_score"] + 1e-4
    assert "trajectory_scores" in res["meta"]
    assert len(res["meta"]["trajectory_scores"]) >= 2


def test_abigx_fallback_expected_ig(make_winclip, images):
    det = make_winclip()
    device = torch.device("cpu")
    # Force non-convergence with impossible criteria
    explainer = ABIGXExplainer(
        detector=det,
        device=device,
        afr_steps=2,
        normal_score_threshold=-1.0,
        min_relative_drop=0.999,
        ig_steps=4,
    )
    
    img = images[:1]
    refs = images[1:].repeat(2, 1, 1, 1)  # [2, 3, 240, 240]
    res = explainer.explain(img, reference_normals=refs)
    
    assert res["meta"]["converged"] is False
    assert res["meta"]["baseline_type"] == "fallback_expected_ig"
    assert res["attribution_map"].shape == (1, 240, 240)

