"""Step 13: unified detector comparison (WinCLIP zero/4-shot, AnomalyCLIP, PatchCore).

Example:
    python scripts/run_baselines.py --detector all --category bottle --max_images 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import append_csv, evaluate, load_yaml, pick, sample_references

from src.detectors.anomalyclip import AnomalyCLIPDetector
from src.detectors.patchcore import PatchCoreDetector
from src.detectors.winclip import WinCLIPDetector
from src.utils.device import get_device
from src.utils.seed import seed_everything


def build(name: str, spec: dict, cfg: dict, cat: str, device):
    t = spec["type"]
    if t == "winclip":
        return WinCLIPDetector(cat, cfg["model"], cfg["pretrained"], device, cfg["image_size"],
                               spec["window_scales"], None, spec["window_chunk_size"],
                               spec["score_mode"], cache_dir=cfg["cache_dir"])
    if t == "anomalyclip":
        return AnomalyCLIPDetector(cfg["model"], cfg["pretrained"], device, spec["score_mode"],
                                   smooth_sigma=spec.get("smooth_sigma", 4.0))
    if t == "patchcore":
        return PatchCoreDetector(device, coreset_ratio=spec["coreset_ratio"],
                                 max_memory=spec["max_memory"], sigma=spec["sigma"],
                                 seed=cfg["seed"])
    raise ValueError(f"unknown detector type {t} for {name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/baselines.yaml")
    p.add_argument("--detector", default="all", choices=["all", "anomalyclip", "patchcore", "winclip"])
    p.add_argument("--category", default="all")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "mps", "cpu", "cuda"])
    a = p.parse_args()

    cfg = load_yaml(a.config)
    device = get_device(a.device or cfg.get("device", "auto"))
    seed_everything(cfg["seed"])
    out = Path(cfg["results_dir"]) / "detector_comparison.csv"
    specs = {n: s for n, s in cfg["detectors"].items() if a.detector in ("all", s["type"])}
    for cat in pick(a.category, cfg["categories"]):
        for name, spec in specs.items():
            det = build(name, spec, cfg, cat, device)
            k = int(spec.get("k_shots", 0))
            det.set_reference(sample_references(cfg["data_root"], cat, k, cfg["seed"],
                                                cfg["image_size"]))
            m = evaluate(det, cfg["data_root"], cat, device, a.max_images, cfg["image_size"])
            row = {"detector": name, "category": cat, "k_shots": k,
                   "image_auroc": m["image_auroc"], "pixel_auroc": m["pixel_auroc"],
                   "pro": m["pro"], "sec_per_image": m["sec_per_image"], "device": str(device)}
            print(row)
            append_csv(out, row, ("detector", "category"))
            del det


if __name__ == "__main__":
    main()
