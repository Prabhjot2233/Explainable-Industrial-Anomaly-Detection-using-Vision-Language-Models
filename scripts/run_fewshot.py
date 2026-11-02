"""Step 11: few-shot WinCLIP+ (k = 0,1,2,4) and prompt-mode ablation.

Example:
    python scripts/run_fewshot.py --category bottle --k_shots all --prompt_mode full_ensemble --max_images 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import append_csv, evaluate, has_row, load_yaml, pick, sample_references

from src.detectors.winclip import WinCLIPDetector
from src.utils.device import get_device
from src.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/fewshot.yaml")
    p.add_argument("--category", default="all")
    p.add_argument("--k_shots", default="all")
    p.add_argument("--prompt_mode", default="all")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cfg = load_yaml(a.config)
    device = get_device(a.device or cfg.get("device", "auto"))
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["results_dir"])
    fs_csv, ab_csv = out_dir / "fewshot_metrics.csv", out_dir / "prompt_ablation.csv"
    cats = pick(a.category, cfg["categories"])
    ks = pick(a.k_shots, cfg["k_shots_list"])
    modes = pick(a.prompt_mode, cfg["prompt_modes"])
    print(f"device={device} categories={cats} k={ks} prompt_modes={modes}")

    for cat in cats:
        for mode in modes:
            det = WinCLIPDetector(
                cat, cfg["model"], cfg["pretrained"], device, cfg["image_size"],
                cfg["window_scales"], None, cfg["window_chunk_size"], cfg["score_mode"],
                mode, alpha_vis=cfg["alpha_vis"], cache_dir=cfg["cache_dir"],
            )
            for k in ks:
                key = {"category": cat, "k_shots": k, "prompt_mode": mode}
                if not a.overwrite and has_row(fs_csv, key):
                    print(f"skip {key} (exists; use --overwrite)")
                    continue
                det.set_reference(sample_references(cfg["data_root"], cat, k, cfg["seed"],
                                                    cfg["image_size"]))
                m = evaluate(det, cfg["data_root"], cat, device, a.max_images, cfg["image_size"])
                row = {**key, **m, "device": str(device)}
                print(row)
                append_csv(fs_csv, row, tuple(key))
                if k == 0:
                    append_csv(ab_csv, row, tuple(key))
            del det


if __name__ == "__main__":
    main()
