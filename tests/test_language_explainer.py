import pytest
import torch
from src.language.vlm_explainer import GroundedLanguageExplainer, LanguageExplanation


def test_language_explainer_normal_guardrail(make_winclip, images):
    det = make_winclip()
    explainer = GroundedLanguageExplainer(det, anomaly_threshold=0.50)

    dummy_attr = torch.zeros(1, 240, 240)
    exp = explainer.explain(
        image=images[:1],
        anomaly_score=0.20,
        attr_map=dummy_attr,
        category="bottle",
    )

    assert isinstance(exp, LanguageExplanation)
    assert exp.is_anomalous is False
    assert exp.predicted_defect_type == "none"
    assert exp.spatial_quadrant == "none"
    assert exp.grounding_bbox == (0, 0, 0, 0)
    assert "PASS" in exp.recommended_action


def test_language_explainer_anomalous_grounding(make_winclip, images):
    det = make_winclip()
    explainer = GroundedLanguageExplainer(det, anomaly_threshold=0.50)

    # Synthetic attribution in top-left
    attr_map = torch.zeros(1, 240, 240)
    attr_map[:, 20:60, 20:60] = 1.0

    exp = explainer.explain(
        image=images[:1],
        anomaly_score=0.85,
        attr_map=attr_map,
        category="carpet",
    )

    assert exp.is_anomalous is True
    assert exp.spatial_quadrant == "top-left"
    assert exp.grounding_bbox[0] >= 15 and exp.grounding_bbox[2] <= 65
    assert exp.grounding_bbox[1] >= 15 and exp.grounding_bbox[3] <= 65
    assert exp.area_percentage > 0.0
    assert exp.predicted_defect_type in ["cut", "hole", "scratch", "contamination", "crack", "deformation"]
    assert exp.defect_confidence > 0.0
    assert "REJECT" in exp.recommended_action or "RE-INSPECT" in exp.recommended_action
    assert isinstance(exp.to_dict(), dict)

