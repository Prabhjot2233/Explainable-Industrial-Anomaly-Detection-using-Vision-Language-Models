"""Step 17: run Saliency, ViT Grad-CAM, IG (black) and Expected-IG on defective images.

Example:
    python scripts/run_visual_explanations.py --category bottle --max_images 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _common import append_csv, load_yaml, sample_references

from src.data.mvtec import MVTecDataset
from src.detectors.winclip import WinCLIPDetector
from src.explainers import IntegratedGradientsExplainer, SaliencyExplainer, ViTGradCAMExplainer
from src.explainers.visualize import save_explanation_comparison_grid
from src.utils.device import get_device
from src.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/explanations_w4.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "mps", "cpu", "cuda"])
    a = p.parse_args()

    cfg = load_yaml(a.config)
    cat = a.category or cfg["category"]
    max_images = a.max_images or cfg.get("max_images", 5)
    device = get_device(a.device or cfg.get("device", "auto"))
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["results_dir"]) / cat

    det = WinCLIPDetector(cat, cfg["model"], cfg["pretrained"], device,
                          window_scales=cfg["window_scales"],
                          window_chunk_size=cfg["window_chunk_size"],
                          score_mode=cfg["score_mode"], cache_dir=cfg["cache_dir"])
    refs = sample_references(cfg["data_root"], cat, cfg["k_shots"], cfg["seed"])
    det.set_reference(refs)
    explainers = {
        "Saliency": SaliencyExplainer(det, device),
        "ViT Grad-CAM": ViTGradCAMExplainer(det, device),
        "IG (Black)": IntegratedGradientsExplainer(det, device, "black", cfg["ig_n_steps"],
                                                   cfg["ig_internal_batch_size"]),
        "Expected-IG (Normal)": IntegratedGradientsExplainer(det, device, "expected",
                                                             cfg["ig_n_steps"],
                                                             cfg["ig_internal_batch_size"]),
    }

    ds = MVTecDataset(cfg["data_root"], cat, split="test")
    defect_idx = [i for i, s in enumerate(ds.samples) if s[2] == 1][:max_images]
    for i in defect_idx:
        item = ds[i]
        img = item["image"][None].to(device)
        stem = f"{item['defect_type']}_{Path(item['path']).stem}"
        import torch

        with torch.no_grad():
            score, amap = det(img)
        maps = {}
        for name, ex in explainers.items():
            out = ex.explain(img, refs)
            maps[name] = out["attribution_map"][0].cpu()
            row = {"category": cat, "image": stem, "explainer": name,
                   "seconds": out["meta"]["seconds"], "device": str(device)}
            append_csv(Path(cfg["results_dir"]) / "runtime_summary.csv", row,
                       ("category", "image", "explainer"))
            print(row)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{stem}.npz", anomaly_map=amap[0].cpu().numpy(),
                            **{k.replace(" ", "_"): v.numpy() for k, v in maps.items()})
        save_explanation_comparison_grid(item["image"], item["mask"], amap[0].cpu(), maps,
                                         out_dir / f"{stem}.png",
                                         f"{cat} / {stem}  score={float(score[0]):.3f}")


if __name__ == "__main__":
    main()
