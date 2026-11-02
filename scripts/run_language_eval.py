"""Step 22: Evaluate Language Explanation Accuracy, Grounding & Hallucination.

Outputs saved to:
  - results/language/language_metrics.csv
  - results/language/predictions.jsonl
  - results/language/human_rating_rubric.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts._common import append_csv, load_yaml, pick, sample_references

from src.data.mvtec import MVTecDataset
from src.detectors.winclip import WinCLIPDetector
from src.explainers.abigx import ABIGXExplainer
from src.language.vlm_explainer import GroundedLanguageExplainer
from src.metrics.language_eval import (
    compute_defect_type_accuracy,
    compute_hallucination_rate,
    compute_region_grounding,
)
from src.utils.device import get_device
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/language.yaml")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cat = args.category
    device = get_device(args.device or "auto")
    seed_everything(0)

    out_dir = Path("results/language")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "language_metrics.csv"
    rubric_csv = out_dir / "human_rating_rubric.csv"
    preds_jsonl = out_dir / "predictions.jsonl"

    print(f"Running Language Explanations Evaluation on {cat} ({device})...")

    det = WinCLIPDetector(
        category=cat,
        model_name=cfg.get("model", "ViT-B-16-plus-240"),
        pretrained=cfg.get("pretrained", "laion400m_e32"),
        device=device,
        score_mode="combined_smooth",
    )
    refs = sample_references(args.data_root, cat, 2, 0, 240)
    det.set_reference(refs)

    explainer = ABIGXExplainer(detector=det, device=device, afr_steps=5, ig_steps=10)
    vlm = GroundedLanguageExplainer(
        detector=det,
        anomaly_threshold=cfg.get("anomaly_threshold", 0.45),
        percentile_threshold=cfg.get("percentile_threshold", 90.0),
    )

    try:
        ds = MVTecDataset(args.data_root, cat, split="test")
        test_items = list(ds)[: args.max_images]
    except Exception as e:
        print(f"Notice: Cannot load dataset ({e}). Generating synthetic evaluation items.")
        g = torch.Generator().manual_seed(42)
        test_items = [
            {"image": torch.randn(3, 240, 240, generator=g), "mask": torch.zeros(1, 240, 240), "label": 0, "defect_type": "good", "path": "synthetic_good.png"},
            {"image": torch.randn(3, 240, 240, generator=g) + 2.0, "mask": torch.ones(1, 240, 240), "label": 1, "defect_type": "cut", "path": "synthetic_cut.png"},
        ]

    predictions = []
    rubric_rows = []

    for item in test_items:
        img_t = item["image"].unsqueeze(0).to(device)
        label = item["label"]
        defect_type = item["defect_type"]
        path_str = Path(item["path"]).name

        with torch.no_grad():
            score_t, _ = det(img_t)
            score_val = float(score_t[0].item())

        exp_res = explainer.explain(img_t, reference_normals=refs)
        attr_map = exp_res["attribution_map"]

        lang_exp = vlm.explain(
            image=img_t,
            anomaly_score=score_val,
            attr_map=attr_map,
            category=cat,
        )

        mask_np = item["mask"].squeeze().cpu().numpy()
        grounding = compute_region_grounding(
            pred_bbox=lang_exp.grounding_bbox,
            pred_quadrant=lang_exp.spatial_quadrant,
            gt_mask=mask_np,
        )

        acc = compute_defect_type_accuracy(
            predicted_defect=lang_exp.predicted_defect_type,
            ground_truth_defect=defect_type,
        )

        pred_record = {
            "category": cat,
            "filename": path_str,
            "ground_truth_label": int(label),
            "ground_truth_defect": defect_type,
            "anomaly_score": score_val,
            "is_gt_normal": (label == 0),
            "predicted_is_anomalous": lang_exp.is_anomalous,
            "predicted_defect": lang_exp.predicted_defect_type,
            "defect_accuracy": acc,
            "predicted_quadrant": lang_exp.spatial_quadrant,
            "quadrant_match": grounding["quadrant_match"],
            "bbox_iou": grounding["bbox_iou"],
            "visual_evidence": lang_exp.visual_evidence,
            "recommended_action": lang_exp.recommended_action,
        }
        predictions.append(pred_record)

        rubric_rows.append({
            "image": path_str,
            "ground_truth_defect": defect_type,
            "predicted_defect": lang_exp.predicted_defect_type,
            "visual_evidence": lang_exp.visual_evidence,
            "clarity_1_to_5": "",
            "groundedness_1_to_5": "",
            "actionability_1_to_5": "",
        })

    # Compute overall rates
    hallucination_stats = compute_hallucination_rate(predictions)
    anom_preds = [p for p in predictions if not p["is_gt_normal"]]
    defect_acc = np.mean([p["defect_accuracy"] for p in anom_preds]) if anom_preds else 1.0
    quad_acc = np.mean([p["quadrant_match"] for p in anom_preds]) if anom_preds else 1.0
    mean_iou = np.mean([p["bbox_iou"] for p in anom_preds]) if anom_preds else 1.0

    summary_row = {
        "category": cat,
        "defect_type_accuracy": float(defect_acc),
        "quadrant_grounding_accuracy": float(quad_acc),
        "mean_bbox_iou": float(mean_iou),
        "normal_false_positive_rate": hallucination_stats["normal_false_positive_rate"],
        "spatial_hallucination_rate": hallucination_stats["spatial_hallucination_rate"],
        "overall_hallucination_score": hallucination_stats["overall_hallucination_score"],
    }
    append_csv(summary_csv, summary_row, ("category",))
    print("Language Evaluation Summary:", summary_row)

    # Save rubric and predictions
    with open(preds_jsonl, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    with open(rubric_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rubric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rubric_rows)

    print(f"Saved evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()

