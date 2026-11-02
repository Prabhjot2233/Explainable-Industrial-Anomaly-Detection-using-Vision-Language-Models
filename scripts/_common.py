"""Shared helpers for runner scripts (config loading, reference sampling, evaluation)."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.mvtec import MVTecDataset  # noqa: E402
from src.metrics.detection import image_auroc, pixel_auroc, pro_score  # noqa: E402


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sample_references(root, category: str, k: int, seed: int, image_size: int = 240):
    """Deterministically pick k normal training images -> [k,3,S,S] (or None if k == 0)."""
    if k <= 0:
        return None
    ds = MVTecDataset(root, category, split="train", image_size=image_size)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    return torch.stack([ds[i]["image"] for i in idx])


def evaluate(detector, root, category: str, device, max_images: int | None = None,
             image_size: int = 240) -> dict:
    ds = MVTecDataset(root, category, split="test", image_size=image_size)
    if max_images:
        # keep both classes: stride through the (sorted) dataset
        step = max(1, len(ds) // max_images)
        ds = torch.utils.data.Subset(ds, list(range(0, len(ds), step))[:max_images])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    labels, scores, masks, maps = [], [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            s, m = detector(batch["image"].to(device))
            labels.append(int(batch["label"][0]))
            scores.append(float(s[0].cpu()))
            masks.append(batch["mask"][0, 0].numpy())
            maps.append(m[0].float().cpu().numpy())
    dt = (time.perf_counter() - t0) / max(len(labels), 1)
    import numpy as np

    masks_np, maps_np = np.stack(masks), np.stack(maps)
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks_np, maps_np),
        "pro": pro_score(masks_np, maps_np),
        "sec_per_image": dt,
        "n_images": len(labels),
    }


def append_csv(path: Path, row: dict, overwrite_keys: tuple[str, ...] | None = None) -> None:
    """Append row; if overwrite_keys given, replace any existing row with matching key values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    if overwrite_keys:
        rows = [r for r in rows if any(str(r.get(k)) != str(row[k]) for k in overwrite_keys)]
    rows.append({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in row.keys()})


def has_row(path: Path, match: dict) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        return any(all(str(r.get(k)) == str(v) for k, v in match.items()) for r in csv.DictReader(f))


def pick(value: str, options: list):
    return list(options) if value == "all" else [type(options[0])(value) if options else value]
